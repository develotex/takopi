# Topics

Topics bind Telegram **forum threads** to a project/branch context. Each topic keeps its own session and default engine, which is ideal for teams or multi-project work.

!!! tip "Workspace workflow"
    If you chose the **workspace** workflow during [onboarding](../tutorials/install.md), topics are already enabled. This guide covers advanced topic configuration and usage.

## Why use topics

- Keep each thread tied to a repo + branch
- Avoid context collisions in busy team chats
- Set a default engine per topic with `/agent set`

## Requirements checklist

- The chat is a **forum-enabled supergroup**
- **Topics are enabled** in the group settings
- To create or rename topics, the bot is an **admin** with **Manage Topics** permission. To use existing topics only, the bot must be a member allowed to post in the group. For ordinary messages (not just commands and replies), disable bot privacy mode via @BotFather; non-admin bots with privacy mode enabled may not receive them.
- If you want topics in project chats, set `projects.<alias>.chat_id`

!!! note "Setting up workspace from scratch"
    If you didn't choose workspace during onboarding and want to enable topics now:

    1. Create a group and enable topics in group settings
    2. Add your bot as admin with "Manage Topics" permission
    3. Update your config to enable topics (see below)

## Enable topics

=== "takopi config"

    ```sh
    takopi config set transports.telegram.topics.enabled true
    takopi config set transports.telegram.topics.scope "auto"
    ```

=== "toml"

    ```toml
    [transports.telegram.topics]
    enabled = true
    scope = "auto" # auto | main | projects | all
    manage_topics = false # optional: use existing GROUP topics without admin rights
    ```

Set `transports.telegram.topics.manage_topics` to `false` to use existing **group** topics only. Sessions, replies, `/new`, and `/ctx set` still work in existing group topics. In groups the bot will not create or rename topics; `/topic` replies with an explanation. Private chats, including creation and renaming of **private topics**, keep their previous behavior. The default is `true` for the usual managed-topic workflow. Startup checks group membership, not whether message delivery or posting works; verify both after changing bot rights/privacy mode.

### Scope explained

- `auto` (default): uses `projects` if any project chats exist, otherwise `main`
- `main`: topics only in the main `chat_id`
- `projects`: topics only in project chats (`projects.<alias>.chat_id`)
- `all`: topics available in both the main chat and project chats

## Create and bind a topic

Run this inside a forum topic thread:

```
/topic <project> @branch
```

Examples:

- In the main chat: `/topic backend @feat/api`
- In a project chat: `/topic @feat/api` (project is implied)

Takopi will bind the topic and rename it to match the context.

## Inspect or change the binding

- `/ctx` shows the current binding
- `/ctx set <project> @branch` updates it
- `/ctx clear` removes it

Note: Outside topics (private chats or main group chats), `/ctx` binds the chat context instead of a topic.

## Reset a topic session

Use `/new` inside the topic to clear stored sessions for that thread.

## Set a default engine per topic

Use `/agent set` inside the topic:

```
/agent set claude
```

## State files

Topic bindings and sessions live in:

- `telegram_topics_state.json`

## Common issues and fixes

- **"topics commands are only available..."**
  - Your `scope` does not include this chat. Update `topics.scope`.
- **"chat is not a supergroup" / "topics enabled but chat does not have topics"**
  - Convert the group to a supergroup and enable topics.
- **"bot lacks manage topics permission"**
  - Grant Manage Topics only if the bot should create or rename topics. Otherwise set `transports.telegram.topics.manage_topics = false` and ensure it can post messages.

## Related

- [Projects and branches](../tutorials/projects-and-branches.md)
- [Route by chat](route-by-chat.md)
- [Chat sessions](chat-sessions.md)
- [Multi-engine workflows](../tutorials/multi-engine.md)
- [Switch engines](switch-engines.md)
