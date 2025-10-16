# UNDOCUMENTED WORK
So much of what I do is done privately and not documented properly for myself.

```{tableofcontents}
```


Here's stuff you need to document:

- SD, flux, etc. etc. IMG GEN
- img+txt --> img
- vid gen
- AUDIO STUFF:
  - STT, **TTS**
    - automatic transcription given a YT url
      - diarized
      - SRT subs
    - translation
    - **voice clone**
      - prosody tricks
  - DATA
    - datasets
    - how to clean data
      - ffmpeg, audacity, etc.
    - how to find the best seed/ref speaker voice
    - how to tweak speaker ref clips to get better outputs
      - fewer artifacts
  - how to structure the text prompt to get emotional speech
  - ASR, voice restoration/clean
    - speaker ID
    - diarization
  - dialogue
    - stop words, natural turns
  - (TTS) dialogue generation from multiple speakers
  - mitigating cursed sound artifacts
  - how to tune sampling and other cfg to help reproduce and get clean shit
  - different models tried (chatterbox, index, vibevoice, etc)
    - example outputs
      - good
      - bad
      - weird
  - different tools tired
    - script
    - notebook
    - comfy
- "AI editors"
  - aider
  - goose
  - void
  - jupylab
  - etc. all those vscode extension ones
- **offline/local model dev**
  - env/deps: CUDA, apt holds, pyenv, deps, hardware, etc.
  - **the setup**
    - open webui
    - n8n
    - oogabooga
    - chainlit
    - why ollama is fuckign garbo
    - vllm good
  - model bins:
    - pytorch pickles
    - safetensors
    - onnx
    - GGUF
    - llamafile
  - model management
    - getting the shit
    - pathing
  - self-hosted
    - tailscale
    - docker-compose
    - all the god-damned BS and torturous debugging to get anythign running
    - reiteration on why ollama is total ass
      - fk ollama
  - working with HF hub..........
    - boy
  - managing data
    - rl ds
    - tune ds
    - pretraining
    - scraping data
    - finding more
    - how the f to store it
    - formats
  - tuning
    - lora
    - axolotl
    - torchtune
  - gpu mgmt stuff (for me, just ez, slick visual and dashboard view)
  - what models suck
  - what models have been surprising
  - managing prompt
- reproducibility
- sampling intuitions
- **certified rippers** (dope ass modles that **JUST WORK**)
  - e.g. SAM, depth-anything, siglip, whisper, FLUX
- **condemned, cursed abominations** (models/methods that are so f'd and needy and assuming and cherry-picked that it feels malicious by authors)
  - .....um, a lot of stuff from modelscope
  - any model that thinks it's coo to tell user to install their BS py pkg to just run god-damned inference witht heir bullshit model
    - fuck off
  - look there's a whole lot of condemned models
- horrendously misinformed and non-expert former fullstack etc. dev now turbo bigbrain deep thonk AI thought leader expert who posts their enlighten thoughts about shit, and then HN and GH eat it the fuck up lol
- how to spot hype, and to accept it into your life






----




Rough idea of buckets or categories (with overlap/dupe):

- EXPERIENCE  🆖: we ended up dumping a bunch of random episodes we commonly encountered in our career, without writing them out or providing basic items/values for the other buckets 💥 Don't do this; it's super lame and not interpretable to anyone but you. Write it out proper next time.
  - research
    - how i got here
    - why i can do what i can do
    - eating shit less
      - how to eat shit more (bad practices)
    - best practices
    - env, setup, tooling
  - mle
    - *largely same as research above (as I am ML research eng)*
    - red flags in ML eng
      - "PoC"
      - "new model"
      - "need to train more"
      - not knowing how to A/B
        - they don't even know what a seed is
      - not knowing what data they are using
      - not looking at the data
        - don't bother with even considering data
          - don't know how to sample
          - don't know basic statistics
          - don't know what the data looks like
      - no visualization
      - no eval
        - eval metrics used are absurdly disjoint from anything related to the data and distributions the model needs to model
        - use toy, often open source datasets with different domain/distrib. and expect this transfers to internal data and application
        - too few test samples
        - no consideration of train/val/test or train/eval split
        - no data curation
        - no consideration of testing strategy
        - no regular testing
        - no sharing of code, data, model, branch, experiment cfg, or run for results that are being presented to engineers and upper management, very often highlighting some vague and non-verifiable "performance improvements" over some previous, equally non-verified and vague "performance" baseline
          - when asked, most common response is confusion
            - do not know what a "seed" is
            - do not know what data was used for the "improvements" or benchmarking
              - often do not actually know, AT ALL, what the data even is, let alone its provenance, format, ontology, version, size, etc.
              - if do "know", it is the assumption of "whatever is in the config or defaults; I made no change"
                - do not know or have even considered whether data may have changed either:
                  - in-place (fix labels, prune bad samples, etc.)
                  - unaware of any changes to ontology, formats, or data versions: "it's what's in master" They cannot speak to which hash or release of master they are referring.
                    - **it's not what's in master. because none of their work for the past 7 months has been based on master**
                      - It's all been done in the individual's (or function's) private branch, which has diverged from master so significantly that their (non-current, or non-tracked) fiefdom branch could not be considered even part of the platform or framework or tooling that needs to be used to collaboratively develop software.
                      - they never bother considering diffs or changes in master that responded to things like:
                        - catastrophic, major bugs in data pipelines, modeling code, loss functions, training schedule, default config, etc.
                      - thus, their "work" and findings and $100K+ cavorts in cloud instance fees for broken models is completely invalidated, and everyone in attendance has been misinformed and misled about progress on your actual job responsibilities
        - nothing changes; there is no need to, nor direction provided by technical "leadership"
          - technical leadership is either so overstretched they could never take the time to do the work for them
          - or, "technical leadership" is not actually "technical".
            - they have no interest, participation, or understanding of any technical work
            - they rarely, if ever, are "boots on the ground" or visit the "trenches"
            - do not know how to engage or respond to someone who is requesting to see the actual work, the code, the artifacts, and experiments
              - have not done before, and don't know where to begin


      - "my own branch"
        - often not shared
        - if shared--as in, there exists a branch in remote--it is never current
        - individuals will go weeks, months, even year detached from master
        - operate in mega branch
        - code cannot be run
          - need dirty changes or
          - need to get this file
          - in general, nothing documented in the source of truth (code)
            - sometimes documented, almost always outdated, elsewhere:
              - confluence
              - google doc
              - slack thread
            - often lack READ permissions to whatever private "source" of "truth" used
      - model-anything
      - refusal to touch things outside of model or fun sexy stuff
        - attempts to delegate this
      - inability to debug issues (including those with "modeling")
      - adjectives like "better", "good", "stronger", "lighter", etc.
        - (instead of easily consumable empirical metrics that communicate this more simply and effectively)
      -
  - dev
  - professional
    - communic
    - collab
    - operating
    - workplace
  -

- PROJECTS
- DEBUGGING
- TOOLS
- LEARNINGS
- ACHIEVEMENTS
