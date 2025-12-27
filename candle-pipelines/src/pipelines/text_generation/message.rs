/// The role of a message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System instructions that guide the model's behavior.
    System,
    /// A message from the user.
    User,
    /// A response from the assistant/model.
    Assistant,
    /// A tool/function result returned to the model.
    Tool,
}

impl Role {
    /// Returns the role as a lowercase string slice.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Message {
    role: Role,
    content: String,
}

impl Message {
    /// Helper to construct a system message.
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
        }
    }

    /// Helper to construct a user message.
    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
        }
    }

    /// Helper to construct an assistant message.
    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
        }
    }

    /// Helper to construct a tool result message.
    pub fn tool(content: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
        }
    }

    /// Returns the message's role.
    pub fn role(&self) -> &Role {
        &self.role
    }

    /// Returns the message content.
    pub fn content(&self) -> &str {
        &self.content
    }
}

/// Extension trait for slices of messages.
#[allow(dead_code)]
pub trait MessageVecExt {
    /// Returns the content of the last user message, if any.
    fn last_user(&self) -> Option<&str>;
    /// Returns the content of the last assistant message, if any.
    fn last_assistant(&self) -> Option<&str>;
    /// Returns the content of the system message, if any.
    fn system(&self) -> Option<&str>;
}

impl<T: AsRef<[Message]>> MessageVecExt for T {
    fn last_user(&self) -> Option<&str> {
        self.as_ref()
            .iter()
            .rev()
            .find(|m| m.role() == &Role::User)
            .map(|m| m.content())
    }

    fn last_assistant(&self) -> Option<&str> {
        self.as_ref()
            .iter()
            .rev()
            .find(|m| m.role() == &Role::Assistant)
            .map(|m| m.content())
    }

    fn system(&self) -> Option<&str> {
        self.as_ref()
            .iter()
            .find(|m| m.role() == &Role::System)
            .map(|m| m.content())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_last_user() {
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("First"),
            Message::assistant("Answer"),
            Message::user("Second"),
        ];
        assert_eq!(messages.last_user(), Some("Second"));
    }

    #[test]
    fn test_last_assistant() {
        let messages = vec![
            Message::user("Q"),
            Message::assistant("A1"),
            Message::assistant("A2"),
        ];
        assert_eq!(messages.last_assistant(), Some("A2"));
    }

    #[test]
    fn test_system() {
        let messages = vec![Message::system("Sys"), Message::user("Q")];
        assert_eq!(messages.system(), Some("Sys"));
    }

    #[test]
    fn test_empty() {
        let messages: Vec<Message> = vec![];
        assert_eq!(messages.last_user(), None);
        assert_eq!(messages.last_assistant(), None);
        assert_eq!(messages.system(), None);
    }
}
