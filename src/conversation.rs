use std::path::Path;

use crate::{Message, Role};

/// Strategy used to trim a conversation when it exceeds the configured context limit.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TruncationStrategy {
    /// Keep all system prompts and the most recent messages.
    KeepSystemAndRecent {
        /// Number of most recent non-system messages to keep.
        recent_messages: usize,
    },
    /// Keep system prompts, the first user turn, and the most recent messages.
    KeepSystemFirstUserAndRecent {
        /// Number of most recent non-system messages to keep.
        recent_messages: usize,
    },
    /// Collapse older turns into a compact assistant summary while keeping the latest ones.
    Summarize {
        /// Maximum token budget for the generated summary.
        max_tokens: usize,
        /// Number of recent non-system messages to preserve verbatim.
        keep_recent: usize,
    },
}

impl Default for TruncationStrategy {
    fn default() -> Self {
        Self::KeepSystemAndRecent {
            recent_messages: 16,
        }
    }
}

/// A first-class container for managing a multi-turn conversation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Conversation {
    messages: Vec<Message>,
    context_limit: Option<usize>,
    truncation_strategy: TruncationStrategy,
}

impl Conversation {
    /// Create an empty conversation.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            context_limit: None,
            truncation_strategy: TruncationStrategy::default(),
        }
    }

    /// Attach a system prompt that will be kept at the start of the conversation.
    pub fn with_system(mut self, content: impl Into<String>) -> Self {
        self.add_system(content);
        self
    }

    /// Apply a context limit. When the estimated token count exceeds this limit, messages
    /// are truncated using the configured strategy.
    pub fn with_context_limit(mut self, limit: usize) -> Self {
        self.set_context_limit(Some(limit));
        self
    }

    /// Set a context limit after creation.
    pub fn set_context_limit(&mut self, limit: Option<usize>) {
        self.context_limit = limit;
        self.enforce_context_limit();
    }

    /// Configure the truncation strategy used when enforcing the context limit.
    pub fn with_truncation_strategy(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation_strategy = strategy;
        self.enforce_context_limit();
        self
    }

    /// Borrow the current message history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Add a user turn to the conversation.
    pub fn add_user(&mut self, content: impl Into<String>) {
        self.messages.push(Message::user(&content.into()));
        self.enforce_context_limit();
    }

    /// Add an assistant turn to the conversation.
    pub fn add_assistant(&mut self, content: impl Into<String>) {
        self.messages.push(Message::assistant(&content.into()));
        self.enforce_context_limit();
    }

    /// Add a system prompt to the conversation.
    pub fn add_system(&mut self, content: impl Into<String>) {
        let system_message = Message::system(&content.into());
        // Replace any existing system prompt to keep the most recent one pinned at the front.
        if let Some(position) = self
            .messages
            .iter()
            .position(|message| matches!(message.role(), Role::System))
        {
            self.messages[position] = system_message;
        } else {
            self.messages.insert(0, system_message);
        }
        self.enforce_context_limit();
    }

    /// Return the configured context limit, if any.
    pub fn context_limit(&self) -> Option<usize> {
        self.context_limit
    }

    /// Estimate the number of tokens used by the current conversation.
    ///
    /// This lightweight heuristic counts whitespace-separated tokens in the message
    /// content and includes a small constant to account for role and formatting.
    pub fn estimated_tokens(&self) -> usize {
        self.messages.iter().map(Self::message_token_cost).sum()
    }

    /// Return the remaining budget if a context limit is set.
    pub fn remaining_budget(&self) -> Option<usize> {
        self.context_limit
            .map(|limit| limit.saturating_sub(self.estimated_tokens()))
    }

    /// Check whether the conversation has enough room for an additional token count.
    pub fn has_budget_for(&self, additional_tokens: usize) -> bool {
        match self.context_limit {
            Some(limit) => self.estimated_tokens().saturating_add(additional_tokens) <= limit,
            None => true,
        }
    }

    /// Create a branched conversation that shares the current history but can evolve independently.
    pub fn branch(&self) -> Self {
        self.clone()
    }

    /// Clear all non-system messages from the conversation, preserving pinned system prompts.
    pub fn clear_history(&mut self) {
        let system_prompts: Vec<Message> = self
            .messages
            .iter()
            .filter(|message| matches!(message.role(), Role::System))
            .cloned()
            .collect();
        self.messages = system_prompts;
    }

    /// Persist the conversation to disk in JSON format.
    pub fn save(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let serialized = serde_json::to_string_pretty(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load a conversation from a JSON file created with [`Conversation::save`].
    pub fn load(path: impl AsRef<Path>) -> crate::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    fn enforce_context_limit(&mut self) {
        let Some(limit) = self.context_limit else {
            return;
        };

        if self.estimated_tokens() <= limit {
            return;
        }

        // Apply the truncation strategy. If the resulting history is still too large,
        // fall back to keeping only the system prompts and the newest non-system messages
        // that fit into the budget.
        self.apply_truncation_strategy();

        if self.estimated_tokens() <= limit {
            return;
        }

        if matches!(
            self.truncation_strategy,
            TruncationStrategy::Summarize { .. }
        ) {
            self.shrink_summary_to_fit(limit);

            if self.estimated_tokens() <= limit {
                return;
            }
        }

        let system_prompts: Vec<Message> = self
            .messages
            .iter()
            .filter(|message| matches!(message.role(), Role::System))
            .cloned()
            .collect();

        let mut retained = system_prompts;
        for message in self
            .messages
            .iter()
            .filter(|message| !matches!(message.role(), Role::System))
        {
            if !self.has_budget_for_message(message, limit, &retained) {
                break;
            }
            retained.push(message.clone());
        }
        self.messages = retained;
    }

    fn has_budget_for_message(
        &self,
        message: &Message,
        limit: usize,
        retained: &[Message],
    ) -> bool {
        let mut temp = retained.to_vec();
        temp.push(message.clone());
        temp.iter().map(Self::message_token_cost).sum::<usize>() <= limit
    }

    fn apply_truncation_strategy(&mut self) {
        match self.truncation_strategy {
            TruncationStrategy::KeepSystemAndRecent { recent_messages } => {
                self.retain_recent(recent_messages);
            }
            TruncationStrategy::KeepSystemFirstUserAndRecent { recent_messages } => {
                self.retain_first_user_and_recent(recent_messages);
            }
            TruncationStrategy::Summarize {
                max_tokens,
                keep_recent,
            } => {
                self.summarize_and_retain(max_tokens, keep_recent);
            }
        }
    }

    fn retain_recent(&mut self, recent_messages: usize) {
        let mut systems: Vec<Message> = self
            .messages
            .iter()
            .filter(|message| matches!(message.role(), Role::System))
            .cloned()
            .collect();

        let mut recents: Vec<Message> = self
            .messages
            .iter()
            .filter(|message| !matches!(message.role(), Role::System))
            .cloned()
            .rev()
            .take(recent_messages)
            .collect();
        recents.reverse();

        systems.append(&mut recents);
        self.messages = systems;
    }

    fn retain_first_user_and_recent(&mut self, recent_messages: usize) {
        let mut systems: Vec<Message> = self
            .messages
            .iter()
            .filter(|message| matches!(message.role(), Role::System))
            .cloned()
            .collect();

        let first_user = self
            .messages
            .iter()
            .find(|message| matches!(message.role(), Role::User))
            .cloned();

        let mut recents: Vec<Message> = self
            .messages
            .iter()
            .filter(|message| !matches!(message.role(), Role::System))
            .cloned()
            .rev()
            .take(recent_messages)
            .collect();
        recents.reverse();

        if let Some(user) = first_user {
            systems.push(user);
        }
        systems.append(&mut recents);
        self.messages = systems;
    }

    fn summarize_and_retain(&mut self, max_tokens: usize, keep_recent: usize) {
        let mut systems: Vec<Message> = self
            .messages
            .iter()
            .filter(|message| matches!(message.role(), Role::System))
            .cloned()
            .collect();

        let non_system: Vec<Message> = self
            .messages
            .iter()
            .filter(|message| !matches!(message.role(), Role::System))
            .cloned()
            .collect();

        if non_system.len() <= keep_recent {
            self.messages = systems;
            self.messages.extend(non_system);
            return;
        }

        let (older, recent) = non_system.split_at(non_system.len() - keep_recent);

        let summary_content = self.compact_summary(older, max_tokens);
        if let Some(summary) = summary_content {
            systems.push(Message::assistant(&summary));
        }

        systems.extend_from_slice(recent);
        self.messages = systems;
    }

    fn compact_summary(&self, messages: &[Message], max_tokens: usize) -> Option<String> {
        if messages.is_empty() || max_tokens == 0 {
            return None;
        }

        let parts: Vec<String> = messages
            .iter()
            .map(|message| {
                let role = message.role().as_str();
                let mut content_tokens: Vec<&str> = message.content().split_whitespace().collect();
                if content_tokens.len() > max_tokens {
                    content_tokens.truncate(max_tokens);
                    content_tokens.push("…");
                }
                let content = content_tokens.join(" ");
                format!("{role}: {content}")
            })
            .collect();

        let mut summary_tokens: Vec<&str> = Vec::new();
        for part in &parts {
            for token in part.split_whitespace() {
                if summary_tokens.len() >= max_tokens {
                    summary_tokens.push("…");
                    let summary = summary_tokens.join(" ");
                    return Some(format!("Summary of earlier messages: {summary}"));
                }
                summary_tokens.push(token);
            }
        }

        if summary_tokens.is_empty() {
            None
        } else {
            Some(format!(
                "Summary of earlier messages: {}",
                summary_tokens.join(" ")
            ))
        }
    }

    fn shrink_summary_to_fit(&mut self, limit: usize) {
        let mut systems: Vec<Message> = self
            .messages
            .iter()
            .filter(|message| matches!(message.role(), Role::System))
            .cloned()
            .collect();

        let mut summary: Option<Message> = None;
        let mut recents: Vec<Message> = Vec::new();

        for message in self
            .messages
            .iter()
            .filter(|message| !matches!(message.role(), Role::System))
        {
            if summary.is_none()
                && matches!(message.role(), Role::Assistant)
                && message
                    .content()
                    .starts_with("Summary of earlier messages:")
            {
                summary = Some(message.clone());
            } else {
                recents.push(message.clone());
            }
        }

        let base_tokens: usize = systems
            .iter()
            .chain(recents.iter())
            .map(Self::message_token_cost)
            .sum();

        let Some(mut summary_message) = summary else {
            return;
        };

        if base_tokens >= limit {
            self.messages = systems;
            self.messages.extend(recents);
            return;
        }

        let summary_prefix_tokens = 4; // "Summary of earlier messages:" -> 4 tokens
        let available_for_summary_content = limit
            .saturating_sub(base_tokens)
            .saturating_sub(summary_prefix_tokens)
            .saturating_sub(1); // Reserve one token for the assistant role.

        if available_for_summary_content == 0 {
            self.messages = systems;
            self.messages.extend(recents);
            return;
        }

        let summary_body = summary_message
            .content()
            .trim_start_matches("Summary of earlier messages:")
            .trim();

        let mut tokens: Vec<&str> = summary_body.split_whitespace().collect();
        if tokens.len() > available_for_summary_content {
            tokens.truncate(available_for_summary_content.saturating_sub(1));
            tokens.push("…");
        }

        let new_content = format!("Summary of earlier messages: {}", tokens.join(" "));
        summary_message = Message::assistant(&new_content);

        systems.push(summary_message);
        systems.extend(recents);
        self.messages = systems;
    }

    fn message_token_cost(message: &Message) -> usize {
        1usize + message.content().split_whitespace().count()
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }
}

impl AsRef<[Message]> for Conversation {
    fn as_ref(&self) -> &[Message] {
        self.messages()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_is_pinned() {
        let mut conversation = Conversation::new().with_system("You are helpful");
        conversation.add_user("Hello");
        conversation.add_system("Stay concise");

        assert_eq!(conversation.messages().len(), 2);
        assert_eq!(conversation.messages()[0].role(), &Role::System);
        assert_eq!(conversation.messages()[0].content(), "Stay concise");
    }

    #[test]
    fn branching_creates_independent_histories() {
        let mut base = Conversation::new().with_system("System");
        base.add_user("Hi");

        let mut branch = base.branch();
        branch.add_assistant("Hello!");

        assert_eq!(base.messages().len(), 2);
        assert_eq!(branch.messages().len(), 3);
    }

    #[test]
    fn enforce_recent_truncation() {
        let mut conversation = Conversation::new()
            .with_system("System")
            .with_context_limit(8)
            .with_truncation_strategy(TruncationStrategy::KeepSystemAndRecent {
                recent_messages: 1,
            });

        conversation.add_user("First question");
        conversation.add_assistant("First answer");
        conversation.add_user("Second question");

        // Only system prompt and the most recent non-system message remain within limit.
        assert_eq!(conversation.messages().len(), 2);
        assert_eq!(conversation.messages()[0].role(), &Role::System);
        assert_eq!(conversation.messages()[1].role(), &Role::User);
        assert_eq!(conversation.messages()[1].content(), "Second question");
    }

    #[test]
    fn summarize_truncation_collapses_history() {
        let mut conversation = Conversation::new()
            .with_system("System")
            .with_context_limit(15)
            .with_truncation_strategy(TruncationStrategy::Summarize {
                max_tokens: 4,
                keep_recent: 1,
            });

        conversation.add_user("First question about Rust");
        conversation.add_assistant("An answer about ownership");
        conversation.add_user("Another follow up");

        let summary_count = conversation
            .messages()
            .iter()
            .filter(|m| matches!(m.role(), Role::Assistant) && m.content().contains("Summary"))
            .count();

        assert_eq!(summary_count, 1);
        assert!(conversation
            .messages()
            .iter()
            .any(|m| m.content().contains("Another follow up")));
    }

    #[test]
    fn save_and_load_round_trip() {
        let mut conversation = Conversation::new()
            .with_system("System")
            .with_context_limit(32);
        conversation.add_user("Hello");
        conversation.add_assistant("Hi there");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conversation.json");

        conversation.save(&path).unwrap();
        let loaded = Conversation::load(&path).unwrap();

        assert_eq!(conversation.messages(), loaded.messages());
        assert_eq!(conversation.context_limit(), loaded.context_limit());
    }
}
