use crate::client::ClientConfig;
use crate::error::{Error, Result};
use crate::types::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};
use crate::utils::{security::create_safe_error_message, validation};
use async_stream::try_stream;
use futures::stream::Stream;
use futures::StreamExt; // Required for .next() on stream
#[cfg(not(target_arch = "wasm32"))] // Only used by native StreamReader path
use futures::TryStreamExt; // Required for .map_err() on stream
use reqwest::Client;
use serde_json;
use std::pin::Pin;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

// Platform-specific imports for async sleep and stream handling
#[cfg(not(target_arch = "wasm32"))]
use tokio::time::sleep;
#[cfg(not(target_arch = "wasm32"))]
use tokio_util::{
    codec::{FramedRead, LinesCodec},
    io::StreamReader,
};

#[cfg(target_arch = "wasm32")]
use gloo_timers::future::TimeoutFuture;
// Bytes is used by reqwest::Body::bytes_stream() which yields Result<Bytes, _>
// No explicit import needed if `bytes::Bytes` is re-exported by `reqwest` or used implicitly.
// Let's add it if compiler complains: `use bytes::Bytes;`


// Streaming safety limits
const MAX_LINE_LENGTH: usize = 64 * 1024; // 64KB per line (UTF-8 bytes)
const MAX_TOTAL_CHUNKS: usize = 10_000; // Maximum successfully parsed ChatCompletionChunk items per stream

// Type alias for the chat stream based on platform
#[cfg(not(target_arch = "wasm32"))]
type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>;
#[cfg(target_arch = "wasm32")]
type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + 'static>>;


pub struct ChatApi {
    pub client: Client,
    pub config: ClientConfig,
}

impl ChatApi {
    pub fn new(client: Client, config: &ClientConfig) -> Self {
        Self {
            client,
            config: config.clone(),
        }
    }

    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        validation::validate_chat_request(&request)?;
        validation::check_token_limits(&request)?;

        let url = self
            .config
            .base_url
            .join("chat/completions")
            .map_err(|e| Error::ApiError {
                code: 400,
                message: format!("Invalid URL: {}", e),
                metadata: None,
            })?;

        let mut retry_count = 0;
        let mut backoff_ms = self.config.retry_config.initial_backoff_ms;

        let response = loop {
            let response_result = self
                .client
                .post(url.clone())
                .headers(self.config.build_headers()?)
                .json(&request)
                .send()
                .await;
            
            let response = match response_result {
                Ok(resp) => resp,
                Err(e) => {
                    // Network or other reqwest error before getting a response
                    // This error might not have a status code in the same way.
                    // We can check if it's a timeout or connection error for retry.
                    // For now, just retry if max_retries not reached.
                    if retry_count < self.config.retry_config.max_retries {
                        retry_count += 1;
                        let log_message = format!(
                            "Retrying request ({}/{}) after {} ms due to connection error: {}",
                            retry_count,
                            self.config.retry_config.max_retries,
                            backoff_ms,
                            e
                        );
                        #[cfg(feature = "tracing")] tracing::warn!("{}", log_message);
                        #[cfg(not(feature = "tracing"))] eprintln!("{}", log_message);

                        #[cfg(not(target_arch = "wasm32"))]
                        sleep(Duration::from_millis(backoff_ms)).await;
                        #[cfg(target_arch = "wasm32")]
                        TimeoutFuture::new(backoff_ms as u32).await;
                        
                        backoff_ms = std::cmp::min(backoff_ms * 2, self.config.retry_config.max_backoff_ms);
                        continue;
                    } else {
                        // Max retries reached for a send error
                        return Err(Error::HttpError(e));
                    }
                }
            };


            let status = response.status();

            if self
                .config
                .retry_config
                .retry_on_status_codes
                .contains(&status.as_u16())
                && retry_count < self.config.retry_config.max_retries
            {
                retry_count += 1;
                
                let log_message = format!(
                    "Retrying request ({}/{}) after {} ms due to status code {}",
                    retry_count,
                    self.config.retry_config.max_retries,
                    backoff_ms,
                    status.as_u16()
                );

                #[cfg(feature = "tracing")]
                tracing::warn!("{}", log_message);
                #[cfg(not(feature = "tracing"))]
                eprintln!("{}", log_message);
                
                #[cfg(not(target_arch = "wasm32"))]
                sleep(Duration::from_millis(backoff_ms)).await;
                #[cfg(target_arch = "wasm32")]
                TimeoutFuture::new(backoff_ms as u32).await;


                backoff_ms = std::cmp::min(backoff_ms * 2, self.config.retry_config.max_backoff_ms);
                continue;
            }
            break response;
        };

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(Error::ApiError {
                code: status.as_u16(),
                message: body.clone(), // body is already a string here
                metadata: None,
            });
        }

        if body.trim().is_empty() {
            return Err(Error::ApiError {
                code: status.as_u16(),
                message: "Empty response body".into(),
                metadata: None,
            });
        }

        let chat_response =
            serde_json::from_str::<ChatCompletionResponse>(&body).map_err(|e| Error::ApiError {
                code: status.as_u16(), // Use original status for consistency
                message: create_safe_error_message(
                    &format!("Failed to decode JSON: {}. Body was: {}", e, body),
                    "Chat completion JSON parsing error",
                ),
                metadata: None,
            })?;

        for choice in &chat_response.choices {
            if let Some(tool_calls) = &choice.message.tool_calls {
                for tc in tool_calls {
                    if tc.kind != "function" {
                        return Err(Error::SchemaValidationError(format!(
                            "Invalid tool call kind: {}. Expected 'function'",
                            tc.kind
                        )));
                    }
                }
            }
        }
        Ok(chat_response)
    }

    pub fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> ChatStream {
        let client = self.client.clone();
        let config = self.config.clone();

        if let Err(e) = validation::validate_chat_request(&request) {
            return Box::pin(futures::stream::once(async { Err(e) }));
        }
        if let Err(e) = validation::check_token_limits(&request) {
            return Box::pin(futures::stream::once(async { Err(e) }));
        }

        let stream = try_stream! {
            let url = config.base_url.join("chat/completions").map_err(|e| Error::ApiError {
                code: 400,
                message: format!("Invalid URL: {}", e),
                metadata: None,
            })?;

            let mut req_body = serde_json::to_value(&request).map_err(|e| Error::ApiError {
                code: 500,
                message: format!("Request serialization error: {}", e),
                metadata: None,
            })?;
            req_body["stream"] = serde_json::Value::Bool(true);

            let response = client
                .post(url)
                .headers(config.build_headers()?)
                .json(&req_body)
                .send()
                .await
                .map_err(Error::HttpError)? // Initial send error
                .error_for_status() // Converts HTTP error statuses (4xx, 5xx) into an Err
                .map_err(|e| { // reqwest::Error from error_for_status
                    // Attempt to get body for more detailed error message
                    // This part is tricky as e.text() would consume the error body if it exists
                    // For now, use e.to_string() which is safer.
                    Error::ApiError {
                        code: e.status().map(|s| s.as_u16()).unwrap_or(500),
                        message: e.to_string(),
                        metadata: None,
                    }
                })?;

            let mut chunk_count = 0usize; // Successfully parsed ChatCompletionChunk items

            #[cfg(not(target_arch = "wasm32"))]
            {
                // Native: Use StreamReader and LinesCodec
                let byte_stream = response.bytes_stream().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
                let stream_reader = StreamReader::new(byte_stream);
                let mut lines = FramedRead::new(stream_reader, LinesCodec::new_with_max_length(MAX_LINE_LENGTH));

                while let Some(line_result) = lines.next().await {
                    let line = line_result.map_err(|e| Error::StreamingError(format!("Failed to read stream line: {}", e)))?;

                    if line.trim().is_empty() {
                        continue;
                    }

                    if line.starts_with("data:") {
                        let data_part = line.trim_start_matches("data:").trim();
                        if data_part == "[DONE]" {
                            break;
                        }

                        chunk_count += 1;
                        if chunk_count > MAX_TOTAL_CHUNKS {
                            Err(Error::StreamingError(format!(
                                "Too many chunks: {} (max: {})",
                                chunk_count, MAX_TOTAL_CHUNKS
                            )))?;
                        }

                        match serde_json::from_str::<ChatCompletionChunk>(data_part) {
                            Ok(chunk) => yield chunk,
                            Err(e) => {
                                let error_msg = create_safe_error_message(
                                    &format!("Failed to parse streaming chunk: {}. Data: {}", e, data_part),
                                    "Streaming chunk parse error (native)"
                                );
                                #[cfg(feature = "tracing")] tracing::error!("{}", error_msg);
                                #[cfg(not(feature = "tracing"))] eprintln!("{}", error_msg);
                                continue;
                            }
                        }
                    } else if line.starts_with(':') {
                        continue; // Ignore SSE comment lines
                    } else {
                         match serde_json::from_str::<ChatCompletionChunk>(&line) {
                            Ok(chunk) => {
                                chunk_count += 1;
                                if chunk_count > MAX_TOTAL_CHUNKS {
                                    Err(Error::StreamingError(format!(
                                        "Too many chunks: {} (max: {})",
                                        chunk_count, MAX_TOTAL_CHUNKS
                                    )))?;
                                }
                                yield chunk
                            },
                            Err(_) => {
                                #[cfg(feature = "tracing")] tracing::warn!("Received non-SSE line that failed to parse (native): {}", line);
                                #[cfg(not(feature = "tracing"))] eprintln!("Received non-SSE line that failed to parse (native): {}", line);
                                continue;
                            }
                        }
                    }
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                // WASM: Manually process byte stream, decode UTF-8, and split lines
                let mut byte_stream = response.bytes_stream();
                let mut byte_buffer: Vec<u8> = Vec::new();

                'stream_loop: while let Some(bytes_result) = byte_stream.next().await {
                    let bytes_chunk = bytes_result.map_err(|e| Error::StreamingError(format!("WASM stream read error: {}", e)))?;
                    byte_buffer.extend_from_slice(&bytes_chunk);

                    loop { 
                        match byte_buffer.iter().position(|&b| b == b'\n') {
                            Some(newline_pos) => {
                                let line_bytes_owned = byte_buffer.drain(..=newline_pos).collect::<Vec<u8>>();
                                
                                let line_str = match String::from_utf8(line_bytes_owned) {
                                    Ok(mut s) => {
                                        s.pop(); 
                                        if s.ends_with('\r') { s.pop(); } 
                                        s
                                    }
                                    Err(e) => {
                                        Err(Error::StreamingError(format!("Invalid UTF-8 line in WASM stream: {}", e)))?;
                                        unreachable!(); 
                                    }
                                };
                                
                                if line_str.len() > MAX_LINE_LENGTH {
                                     Err(Error::StreamingError(format!(
                                        "Line too long (WASM): {} bytes (max: {})",
                                        line_str.len(), MAX_LINE_LENGTH
                                    )))?;
                                }

                                if line_str.trim().is_empty() {
                                    continue;
                                }

                                if line_str.starts_with("data:") {
                                    let data_part = line_str.trim_start_matches("data:").trim();
                                    if data_part == "[DONE]" {
                                        if !byte_buffer.is_empty() {
                                            let remainder = String::from_utf8_lossy(&byte_buffer);
                                            #[cfg(feature = "tracing")] tracing::warn!("Data remaining in WASM buffer after [DONE]: {}", remainder);
                                            #[cfg(not(feature = "tracing"))] eprintln!("Data remaining in WASM buffer after [DONE]: {}", remainder);
                                        }
                                        break 'stream_loop;
                                    }

                                    chunk_count += 1;
                                    if chunk_count > MAX_TOTAL_CHUNKS {
                                        Err(Error::StreamingError(format!(
                                            "Too many chunks (WASM): {} (max: {})",
                                            chunk_count, MAX_TOTAL_CHUNKS
                                        )))?;
                                    }

                                    match serde_json::from_str::<ChatCompletionChunk>(data_part) {
                                        Ok(chunk) => yield chunk,
                                        Err(e) => {
                                            let error_msg = create_safe_error_message(
                                                &format!("Failed to parse streaming chunk: {}. Data: {}", e, data_part),
                                                "Streaming chunk parse error (WASM)"
                                            );
                                            #[cfg(feature = "tracing")] tracing::error!("{}", error_msg);
                                            #[cfg(not(feature = "tracing"))] eprintln!("{}", error_msg);
                                            continue;
                                        }
                                    }
                                } else if line_str.starts_with(':') {
                                    continue; 
                                } else {
                                    match serde_json::from_str::<ChatCompletionChunk>(&line_str) {
                                        Ok(chunk) => {
                                            chunk_count += 1;
                                            if chunk_count > MAX_TOTAL_CHUNKS {
                                                Err(Error::StreamingError(format!(
                                                    "Too many chunks (WASM): {} (max: {})",
                                                    chunk_count, MAX_TOTAL_CHUNKS
                                                )))?;
                                            }
                                            yield chunk
                                        },
                                        Err(_) => {
                                            #[cfg(feature = "tracing")] tracing::warn!("Received non-SSE line that failed to parse (WASM): {}", line_str);
                                            #[cfg(not(feature = "tracing"))] eprintln!("Received non-SSE line that failed to parse (WASM): {}", line_str);
                                            continue;
                                        }
                                    }
                                }
                            }
                            None => break, 
                        }
                    }
                }
                if !byte_buffer.is_empty() {
                    let line_str = String::from_utf8_lossy(&byte_buffer).to_string();
                    if !line_str.trim().is_empty() { 
                         #[cfg(feature = "tracing")]
                         tracing::warn!("Incomplete data at end of WASM stream: {}", line_str);
                         #[cfg(not(feature = "tracing"))]
                         eprintln!("Incomplete data at end of WASM stream: {}", line_str);
                    }
                }
            }
        };
        Box::pin(stream)
    }

    pub async fn simple_completion(&self, model: &str, user_message: &str) -> Result<String> {
        let request = ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![crate::types::chat::Message {
                role: "user".to_string(),
                content: user_message.to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: None,
            response_format: None,
            tools: None,
            provider: None,
            models: None,
            transforms: None,
        };

        let response = self.chat_completion(request).await?;

        response.choices.first()
            .and_then(|choice| Some(choice.message.content.clone()))
            .ok_or_else(|| Error::ApiError {
                code: 500, // Or a more specific error code/type
                message: "No content found in the first choice of the response".into(),
                metadata: None,
            })
    }
}