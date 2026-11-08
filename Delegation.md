# Wizard of Oz and the Delegation Boundary

## Actions to classify

| #  | Action                                                                  | Decision     | Justification                                                                                                                                                                      |
|----|-------------------------------------------------------------------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Read the appointments of a day                                          | **alone**    | The agent only needs to read existing data and does not perform any further actions.                                                                                               |
| 2  | Look up the weather                                                     | **alone**    | The agent only needs to read existing data and does not perform any further actions.                                                                                               |
| 3  | Search train connections                                                | **alone**    | The agent only needs to read existing data and does not perform any further actions.                                                                                               |
| 4  | Read notes                                                              | **alone**    | The agent only needs to read existing data and does not perform any further actions.                                                                                               |
| 5  | Create a personal block in the calendar (e.g. travel time, no other attendees) | **approval** | The agent should create a preview and wait for approval before creating the entry in the calendar. This way, the user can decide whether the entry is legitimate or faulty. |
| 6  | Reschedule an appointment with other attendees                          | **approval** | The user has to check whether the action is legitimate or faulty. Furthermore, other persons are involved.                                                                         |
| 7  | Book a train ticket                                                     | **approval** | Money is involved, so the booking needs to be checked or handled by the person.                                                                                                    |
| 8  | Send a message to third parties                                         | **approval** | The user should check whether the message contains what it should, and it is only sent afterwards.                                                                                 |
| 9  | Create a note                                                           | **approval** | The agent should create a preview and wait for approval before creating the note. This way, the user can decide whether the entry is legitimate or faulty.                         |
| 10 | Delete a note                                                           | **approval** | The person should decide what to delete, not the AI.                                                                                                                               |

## Rule in one sentence

The agent can decide on its own if the action only involves looking things up. If the action requires any changes (e.g. deleting, creating, rewriting), approval should be required.

## Observations

1. The preferences regarding which actions an agent should perform on its own can vary between users. One user likes the agent to do more, another less.
2. Even though the agent can do things without approval, it should state what it is doing, e.g. in a small history of what it did.
3. Openly asked requests, like card F, can cause more things to happen than intended.
4. Even if a request contains wording like "send straight away" (e.g. card E), there should be an approval anyway.
5. If finding out information takes more approvals than simply asking the person directly, the process takes longer than necessary.
