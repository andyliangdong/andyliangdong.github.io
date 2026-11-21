+++
title = 'About'
ShowToc = false
ShowReadingTime = false
+++

I'm Liang Dong, a software engineer with a PhD in Computer Science, researching
reinforcement learning for language models. I reproduce recent papers with controlled experiments
to find out what actually drives their results.

My focus is **RL with verifiable rewards for code generation**, and in particular where the verifier
and the task come apart:

- **Training:** GRPO and its variants, reward design, and what breaks when you change one knob.
- **Reward hacking:** how weak tests, leaky reward suites, and imperfect monitors get exploited, and
  how to measure that honestly.
- **Evaluation:** confidence intervals that resample tasks rather than samples, and audits that
  check the checker.

Each post is about a five-minute read: one question, one experiment, the numbers, and the caveats.
Code and configs for every post are in
[andyliangdong/llm-repro](https://github.com/andyliangdong/llm-repro).

Everything runs on my own hardware (one RTX 3080 Ti, sometimes a rented GPU) on my own time. Models
are small, usually 0.6B–1.5B parameters, so read results as single-run observations, not
benchmarks.

Views are my own and do not represent any employer. Code and text are licensed under
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

- GitHub: [andyliangdong](https://github.com/andyliangdong)
