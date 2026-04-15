use async_trait::async_trait;

use crate::{AgentContext, AgentError};

#[async_trait]
pub trait Agent: Send + Sync {
    type Input: Send + Sync + 'static;
    type Output: Send + Sync + 'static;

    async fn run(
        &self,
        input: Self::Input,
        ctx: &mut AgentContext,
    ) -> Result<Self::Output, AgentError>;
}
