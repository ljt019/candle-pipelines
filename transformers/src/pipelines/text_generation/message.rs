#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Message {
    role: Role,
    content: String,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
        }
    }

    pub fn role(&self) -> &Role {
        &self.role
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[allow(dead_code)]
pub trait MessageVecExt {
    fn last_user(&self) -> Option<&str>;
    fn last_assistant(&self) -> Option<&str>;
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
