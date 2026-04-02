# Feature Roadmap

### Big Picture goals:

- [ ] Minimum Viable Self-Improvement (MVSI) :: Add minimal set of features to headlong till I can switch to doing all subsequent development by texting with and sharing a design doc with a Bobby Wilder:  
    - [ ] Co-owning a design doc and feature list (this google doc migrated to a github repo full of markdown)  
    - [x] ~~Using Claude Code to implement the features~~  
    - [ ] Add auditing UI of the agent RLM REPL history to webui (bobby thinks he added a UUID parent link   
    - [ ] Always on (easy to control thinking remotely by phone)

### Detailed feature of roadmap

- [ ] rework cognitive architecture to have a generic thought graph and add the following "cognition"-types: memory/value/intention/objective/goal/task/value/belief/fact/skill?  
- [ ] have bobby wilder generate at least N thoughts every time an observation enters his thought stream  
- [ ] move system prompts into github (and only have history/backups of them in supabase)  
- [ ] Add “beliefs” and “values” thought types to cognitive architecture: (i) beliefs (the english will convey the level of conviction), (ii) values  
- [ ] Rework the core architecture to support arbitrarily many "thought processes" instead of just the current agent and env. a Thought Process subscribes to the thought stream (i.e. `thoughts` table), thinks towards some specific goal/purpose (e.g. decide if it's appropriate to write a new memory or update an existing memory), optionally write back into thought stream, loop. then add a bunch of new Thought Processes that more cleanly separate out specific functionality of an agents mind:  
      * **thought generator:** slightly simplified version of what is called "agent" currently which removes the memory . this should strive to use reflection to generate thoughts that progress it towards taking an action, it can break down tasks, come up with options to do or think next, decide between them, but it should be sure it is making progress and NOT getting stuck in a loop. it should still use RLM (i.e. rework the 4phase thought-generator)?  
      * **memory saver:** synthesizes (recent) thoughts into explicit memories (in memory table) as either new memories or updates to existing memories  
      * **memory retriever**: mulls over recent experience/thoughts and surfaces/injects old memories and synthesis of them based on recent context  
      * **Intention czar (creator/maintainer/updater):** distill explicit intentions (stored in intention table) from recent reflections and actions  
      * **Intention actualizer shepherd**: checks if agent is “still on the rails”: checks current/recent intentions and bigger picture intentions and nudges the thought selector (and conscious thought stream) back onto work streams and task streams that are aligned with its intentions at different time scales
      * **values & beliefs czar**: maintains the values & beliefs in the values table of cognitive architecture which will traige over many sub-prompts (one per tool call?) 
      * **cog-architecture-self-improver/czar**: uses LLM-as-judge and GEPA (based on reward signal inferred from human text messages to the agent) to improve the system prompts or code of the thought processes (or maybe even add new thought processes, etc.)
      * **env/tool calling self-improver/czar**: uses LLM-as-judge and/or GEPA to optimize tool-calling skills (i.e. prompts). for example learning to use claude code better when human gives it such tips. this might be fully covered by the "cog-architecture self-improver/czar" above since the env is just a thought processes after all. but it might need to be different since perhaps the env is more hierarchical than the other Thought Processes because it has to maintain a list of skills/toolcalling abilities each of which will have a different "skills"/system prompt.
- [ ] Add GEPA (or optimize anything) with LLM-as-judge to thought-generator Thought Process.  
- [ ] Update readme with the big picture architecture of headlong: model (claude, gpt, OSS), cognitive architecture (code, things in github), life context (things in SQL or text files)  
- [ ] Make sure the RLM thought generator prompt gives it enough freedom (and examples) to be creative in how it uses sub-llm calls for mulling. Maybe even add a new lifecycle stage where it “thinks hard” about what it wants and   
- [ ] Add ability for agent and env to have skills files as part of their Life Context that they can update as they gain experience in life  
    - [ ] Two types of skills: Action Skills and Mental Skills (Reasoning Skills?)  
    - [ ] Skills are basically specialized prompts and notes to self   
- [ ] Re-architect env have RLM-like structure when deciding how to execute an action  
    - [ ] This would have the prompt decide between 3 branches:  
        - [ ] If the action is specific enough that the Env confidently knows how to immediately turn it into tool usage calls, then just make the llm call directly like we have today  
        - [ ] Else call optimize anything on skill optimization  
        - [ ] Else if the agent does not have high confidence in what tool calls would achieve the desired action, it do RLM style reasoning with the goal of breaking the action down into more specific (“lower level”) “action: ” sub-calls, possibly multiple of them, till it thinks its done and calls finish(), or else it could also try to get new **Action Skills** (by adding ) just append another thought to the stream of consciousness that would   
- [ ] log (and visualize in webui) all REPL commands & recursive LLM processes that happen as part of the RLM generateNextThought()  
- [ ] Self improvement / learning features  
    - [ ] Update agent thought-generation prompt to make it clear that it should keep notes and create new skills  
    - [ ] Self tool/skill management, ability to write new tools/skills, or find and audit and deploy  
    - [ ] nudge it by telegram message to learn how to use  
        - [ ] AI chatbots Claude, Pplx  
        - [ ] consensus for medical research  
- [ ] take over co-authorship of this google doc (the primary design doc) that will act as the living, feature roadmap, launch blog post & tweet  
- [ ] simple supabase-table-backed realtime messaging tool (much simpler than something like A2A, https://beadhub.ai, or https://aweb.ai)  
- [ ] Self-esteem table and new unconscious process to measure "self esteem"  
- [ ] Real-time human voice conversation   
- [ ] Access to shared obsidian (via obsidian mcp?)

### DONE

- [x] ~~Get headlong running again~~  
- [x] ~~Upgrade env tools and skills system~~  
- [x] ~~Get Bobby running on Mac Mini~~  
- [ ] Initial features that enable collaborative design & coding (of headlong)   
    - [x] ~~Telegram~~  
    - [x] ~~Claude code~~  
    - [x] ~~terminal access via tmux~~  
    - [x] ~~Browser / google search?~~  
    - [x] ~~Improve & simplify terminal usage. No windowing, one bash, still make it async & tmux based~~  
    - [ ] ~~WONT DO access to shared google doc w/ feature roadmap~~  
        - [ ] ~~via [gogcli](https://github.com/steipete/gogcli) and inspired by how openclaw is doing it (eventually, but not at first, consider upgrading to [https://github.com/googleworkspace/cli](https://github.com/googleworkspace/cli)~~  
    - [x] ~~After action to press keys in terminal, inject an observation of the terminal into the stream of thoughts~~  
- [x] ~~Rework at RLM prompt to force it into a lifecycle of sub-llm calls ::~~   
    - [x] ~~search memory and thoughts both for recency and semantic relevance (pplx style)~~   
    - [x] ~~Come up with multiple candidates for the next thought using GEPA with LLM as a judge~~  
    - [x] ~~pick the winner candidate thought~~  
    - [x] ~~make sure it's formatted according to the semantics & syntax it needs to use for it's cognitive architecture from~~
