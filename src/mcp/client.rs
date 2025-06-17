//! MCP client implementation for connecting to MCP servers.

use futures::lock::Mutex; // Changed from tokio::sync::Mutex
use url::Url;

use crate::error::{Error, Result};
use crate::mcp::types::*;

/// MCP client for connecting to and interacting with MCP servers.
pub struct MCPClient {
    /// The HTTP client for making requests
    client: reqwest::Client,
    /// The base URL of the MCP server
    server_url: Url,
    /// Server capabilities once initialized
    capabilities: Mutex<Option<ServerCapabilities>>,
}

impl MCPClient {
    /// Create a new MCP client for the given server URL.
    pub fn new(server_url: impl AsRef<str>) -> Result<Self> {
        let server_url = Url::parse(server_url.as_ref())
            .map_err(|e| Error::ConfigError(format!("Invalid server URL: {}", e)))?;

        Ok(Self {
            client: reqwest::Client::new(),
            server_url,
            capabilities: Mutex::new(None),
        })
    }

    /// Generate a simple request ID
    fn generate_id() -> String {
        // Use a simple timestamp-based ID instead of UUID for now,
        // or switch to a proper UUID if platform-specific features are sorted.
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("req-{}", timestamp)
    }

    /// Initialize the connection to the MCP server.
    pub async fn initialize(
        &self,
        client_capabilities: ClientCapabilities,
    ) -> Result<ServerCapabilities> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Self::generate_id(),
            method: "initialize".to_string(),
            params: Some(
                serde_json::to_value(InitializeParams {
                    capabilities: client_capabilities,
                })
                .map_err(Error::SerializationError)?,
            ),
        };

        let response = self.send_request(request).await?;
        let capabilities = self.parse_response::<ServerCapabilities>(response)?;

        // Store the server capabilities
        let mut caps = self.capabilities.lock().await;
        *caps = Some(capabilities.clone());

        Ok(capabilities)
    }

    /// Get a resource from the server.
    pub async fn get_resource(&self, params: GetResourceParams) -> Result<ResourceResponse> {
        // Check if initialized
        self.ensure_initialized().await?;

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Self::generate_id(),
            method: "getResource".to_string(),
            params: Some(serde_json::to_value(params).map_err(Error::SerializationError)?),
        };

        let response = self.send_request(request).await?;
        self.parse_response::<ResourceResponse>(response)
    }

    /// Call a tool on the server.
    pub async fn tool_call(&self, params: ToolCallParams) -> Result<ToolCallResponse> {
        // Check if initialized
        self.ensure_initialized().await?;

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Self::generate_id(),
            method: "toolCall".to_string(),
            params: Some(serde_json::to_value(params).map_err(Error::SerializationError)?),
        };

        let response = self.send_request(request).await?;
        self.parse_response::<ToolCallResponse>(response)
    }

    /// Execute a prompt on the server.
    pub async fn execute_prompt(
        &self,
        params: ExecutePromptParams,
    ) -> Result<ExecutePromptResponse> {
        // Check if initialized
        self.ensure_initialized().await?;

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Self::generate_id(),
            method: "executePrompt".to_string(),
            params: Some(serde_json::to_value(params).map_err(Error::SerializationError)?),
        };

        let response = self.send_request(request).await?;
        self.parse_response::<ExecutePromptResponse>(response)
    }

    /// Send a sampling response to the server.
    pub async fn respond_to_sampling(&self, id: String, result: SamplingResponse) -> Result<()> {
        // Check if initialized
        self.ensure_initialized().await?;

        let response_payload = JsonRpcResponse { // Renamed to avoid conflict if this was intended to be a request
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(serde_json::to_value(result).map_err(Error::SerializationError)?),
            error: None,
        };

        self.send_response_payload(response_payload).await // Changed to send_response_payload
    }

    /// Get the server capabilities.
    pub async fn capabilities(&self) -> Option<ServerCapabilities> {
        self.capabilities.lock().await.clone()
    }

    /// Send a JSON-RPC request to the server.
    async fn send_request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        let response = self
            .client
            .post(self.server_url.clone())
            .json(&request)
            .send()
            .await
            .map_err(Error::HttpError)?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(Error::ApiError {
                code: status.as_u16(),
                message: text,
                metadata: None,
            });
        }

        let response_body = response.text().await.map_err(Error::HttpError)?;
        serde_json::from_str(&response_body).map_err(|e| {
            Error::SerializationError(e)
        })
    }

    /// Send a JSON-RPC response payload to the server.
    async fn send_response_payload(&self, payload: JsonRpcResponse) -> Result<()> { // Renamed from send_response
        let res = self // Renamed to avoid conflict
            .client
            .post(self.server_url.clone())
            .json(&payload)
            .send()
            .await
            .map_err(Error::HttpError)?;

        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.unwrap_or_default();
             return Err(Error::ApiError {
                code: status.as_u16(),
                message: text,
                metadata: None, // Or some other way to get metadata if available
            });
        }
        Ok(())
    }

    /// Parse a JSON-RPC response into the expected type.
    fn parse_response<T: serde::de::DeserializeOwned>(
        &self,
        response: JsonRpcResponse,
    ) -> Result<T> {
        // Check for errors
        if let Some(error) = response.error {
            return Err(Error::ApiError {
                code: error.code as u16, // Assuming JsonRpcError has a u16 or similar code
                message: error.message,
                metadata: error.data,
            });
        }

        // Parse the result
        match response.result {
            Some(result) => serde_json::from_value(result).map_err(Error::SerializationError),
            None => Err(Error::ConfigError("Response contains no result".into())),
        }
    }

    /// Ensure the client has been initialized.
    async fn ensure_initialized(&self) -> Result<()> {
        if self.capabilities.lock().await.is_none() {
            return Err(Error::ConfigError("MCP client not initialized".into()));
        }
        Ok(())
    }
}