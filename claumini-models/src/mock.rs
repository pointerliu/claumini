use async_trait::async_trait;
use claumini_core::{
    ModelProvider, ModelRequest, ModelResponse, ProviderCapabilities, ProviderError,
};

#[derive(Debug, Clone)]
pub struct MockProvider {
    response: ModelResponse,
    capabilities: ProviderCapabilities,
}

impl MockProvider {
    pub fn new(response: ModelResponse) -> Self {
        Self {
            response,
            capabilities: ProviderCapabilities::default(),
        }
    }

    pub fn new_text(text: impl Into<String>) -> Self {
        Self::new(ModelResponse::text(text))
    }

    pub fn with_capabilities(mut self, capabilities: ProviderCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }
}

#[async_trait]
impl ModelProvider for MockProvider {
    async fn complete(&self, _request: ModelRequest) -> Result<ModelResponse, ProviderError> {
        Ok(self.response.clone())
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities
    }
}
