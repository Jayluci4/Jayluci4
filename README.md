# Hey, I'm Jayant 👋

**AI Engineer & Educator** | Building Neural Networks from Scratch | Making Deep Learning Accessible

I **teach AI and machine learning by implementing algorithms from first principles**. Learn how neural networks, transformers, LSTMs, and modern LLMs actually work - no PyTorch, no TensorFlow, just pure NumPy and clear explanations.

Perfect for:
- **Students** learning deep learning fundamentals
- **Engineers** understanding AI architectures deeply
- **Researchers** exploring neural network implementations
- **Anyone** curious about how ChatGPT, GPT-4, and Stable Diffusion work under the hood

Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) and the philosophy of learning by building.

**Keywords:** *AI from scratch, neural networks tutorial, LSTM tutorial, transformer implementation, RNN backpropagation, word embeddings, sequence to sequence, attention mechanism, deep learning education, machine learning fundamentals, NumPy neural networks*

---

<div align="center">

[![GitHub followers](https://img.shields.io/github/followers/Jayluci4?label=Follow&style=social)](https://github.com/Jayluci4)
[![GitHub stars](https://img.shields.io/github/stars/Jayluci4?style=social)](https://github.com/Jayluci4)
![Profile Views](https://komarev.com/ghpvc/?username=Jayluci4&color=brightgreen)

**14 Repositories** | **~3050 Lines of Educational Code** | **From RNNs to LLMs**

[🚀 Start Learning](#-build-ai-from-scratch-series) | [📄 Research Papers](#-foundational-research-papers) | [📚 Learning Path](#-how-to-learn-ai-from-scratch-complete-roadmap) | [🤝 Collaborate](#-connect--collaborate)

</div>

---

## 🎓 Build AI From Scratch Series

Learning modern AI by implementing the fundamentals in minimal code.

### Part 1: Foundational Components (1990s-2010s)

The building blocks that bridge micrograd and modern transformers:

<table>
<tr>
<th>Repo</th>
<th>Lines</th>
<th>Concept</th>
<th>What You'll Learn</th>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-rnn"><b>micro-rnn</b></a></td>
<td>~350</td>
<td>Recurrent Neural Networks</td>
<td>How neural networks process sequences (vanishing gradients problem)</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-lstm"><b>micro-lstm</b></a></td>
<td>~450</td>
<td>Long Short-Term Memory</td>
<td>How gates solve vanishing gradients (the breakthrough of 1997)</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-gru"><b>micro-gru</b></a></td>
<td>~300</td>
<td>Gated Recurrent Unit</td>
<td>Simplified LSTM with 25% fewer parameters (2014 innovation)</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-embedding"><b>micro-embedding</b></a></td>
<td>~300</td>
<td>Word2Vec Embeddings</td>
<td>How words become vectors (king - man + woman = queen)</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-seq2seq"><b>micro-seq2seq</b></a></td>
<td>~450</td>
<td>Encoder-Decoder</td>
<td>How translation works (and why it needs attention)</td>
</tr>
</table>

**Subtotal: 5 repos, ~1850 lines** - Start here after micrograd!

### Part 2: Modern AI Architectures (2017-2025)

The cutting-edge techniques powering today's AI:

<table>
<tr>
<th>Repo</th>
<th>Lines</th>
<th>Concept</th>
<th>What You'll Learn</th>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-tokenizer"><b>micro-tokenizer</b></a></td>
<td>~80</td>
<td>BPE Tokenization</td>
<td>How text becomes tokens (the first step for all LLMs)</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-attention"><b>micro-attention</b></a></td>
<td>~50</td>
<td>Attention Mechanism</td>
<td>How transformers attend to sequences (powers GPT, BERT)</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-transformer"><b>micro-transformer</b></a></td>
<td>~200</td>
<td>Complete Transformers</td>
<td>How GPT and BERT actually work (putting it all together)</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-rag"><b>micro-rag</b></a></td>
<td>~150</td>
<td>RAG</td>
<td>How ChatGPT answers questions about your documents</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-agent"><b>micro-agent</b></a></td>
<td>~100</td>
<td>ReAct Agents</td>
<td>How autonomous AI systems think and act</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-kg"><b>micro-kg</b></a></td>
<td>~150</td>
<td>Knowledge Graphs</td>
<td>How Google structures knowledge for AI reasoning</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-rlhf"><b>micro-rlhf</b></a></td>
<td>~150</td>
<td>RLHF Alignment</td>
<td>How ChatGPT becomes helpful, harmless, and honest</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-diffusion"><b>micro-diffusion</b></a></td>
<td>~200</td>
<td>Diffusion Models</td>
<td>How Stable Diffusion generates images from noise</td>
</tr>
<tr>
<td><a href="https://github.com/Jayluci4/micro-lora"><b>micro-lora</b></a></td>
<td>~120</td>
<td>Efficient Fine-tuning</td>
<td>How to fine-tune GPT on a laptop (1000x more efficient)</td>
</tr>
</table>

**Subtotal: 9 repos, ~1200 lines**

---

**Grand Total: 14 repos, ~3050 lines teaching AI from RNNs to modern LLMs**

Each repo includes:
- Minimal, readable code (no frameworks)
- Beautiful visualizations
- Interactive Jupyter notebooks
- Step-by-step tutorials

---

## 📄 Foundational Research Papers

Learn the theory behind each implementation. All papers with verified working links:

### Recurrent Neural Networks & Sequence Models
- **[Long Short-Term Memory (1997)](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory)** - Hochreiter & Schmidhuber
  - *The breakthrough that solved vanishing gradients in RNNs*
- **[Learning Phrase Representations using RNN Encoder-Decoder (2014)](https://arxiv.org/abs/1406.1078)** - Cho et al.
  - *Introduced GRU (Gated Recurrent Unit)*
- **[Sequence to Sequence Learning with Neural Networks (2014)](https://arxiv.org/abs/1409.3215)** - Sutskever, Vinyals & Le
  - *LSTM-based encoder-decoder for translation*

### Word Embeddings & Representations
- **[Efficient Estimation of Word Representations in Vector Space (2013)](https://arxiv.org/abs/1301.3781)** - Mikolov et al.
  - *Word2Vec: Skip-gram and CBOW architectures*

### Attention & Transformers
- **[Neural Machine Translation by Jointly Learning to Align and Translate (2015)](https://arxiv.org/abs/1409.0473)** - Bahdanau, Cho & Bengio
  - *First attention mechanism for seq2seq*
- **[Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)** - Vaswani et al.
  - *The transformer architecture that revolutionized NLP*

### Modern AI Techniques
- **[Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239)** - Ho, Jain & Abbeel
  - *Foundation of Stable Diffusion and DALL-E*
- **[LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685)** - Hu et al.
  - *Efficient fine-tuning for LLMs*

### Optimization
- **[Adam: A Method for Stochastic Optimization (2014)](https://arxiv.org/abs/1412.6980)** - Kingma & Ba
  - *The most widely used optimizer in deep learning*

---

## 🎯 Why Build AI from Scratch?

### The Problem with Most Deep Learning Tutorials

Most AI and machine learning tutorials either:
- **Use high-level frameworks** (PyTorch, TensorFlow, Keras) - hiding how backpropagation, gradient descent, and neural networks actually work
- **Are too mathematical** (dense papers, complex notation) - losing intuition and practical understanding

You can use ChatGPT without understanding transformers. But **you can't build the next breakthrough without understanding the fundamentals**.

### The Solution: First Principles Learning

**I implement every algorithm from scratch** - no black boxes, no magic. Just NumPy arrays, clear code, and extensive comments explaining the "why" behind each line.

When you see exactly how:
- LSTM gates solve vanishing gradients
- Attention mechanisms look at all input tokens
- Backpropagation through time works in RNNs
- Word embeddings capture semantic meaning

...you gain **deep intuition** that makes you a better AI engineer, researcher, or practitioner.

### Learning Path: From Basics to State-of-the-Art

```
micrograd (Karpathy) → Foundational Components → Modern Architectures
    ↓                         ↓                           ↓
Backpropagation          RNN, LSTM, GRU            Transformers, RAG
Gradient Descent         Word Embeddings           Diffusion Models
Neural Networks          Seq2Seq                   RLHF, LoRA
```

**Goal:** Democratize AI education. Make cutting-edge concepts accessible to anyone willing to learn.

---

## 📚 Other Educational Projects

**Active Learning Resources:**
- [**BatchNorm**](https://github.com/Jayluci4/BatchNorm) - Understand batch normalization intuitively
- [**ZeroToHeroRL**](https://github.com/Jayluci4/ZeroToHeroRL) - Reinforcement learning examples and tutorials

---

## 🛠️ Production Projects

**AI Applications & Tools:**
- [**mandrake-agent**](https://github.com/Jayluci4/mandrake-agent) - Intelligent agent framework for autonomous task execution
- [**kg-chatbot**](https://github.com/Jayluci4/kg-chatbot) - Knowledge graph-powered conversational AI
- [**kg-eval**](https://github.com/Jayluci4/kg-eval) - Evaluation framework for knowledge graph quality
- [**ultimate-agentic-app**](https://github.com/Jayluci4/ultimate-agentic-app) - Advanced agentic application with multi-agent orchestration
- [**news**](https://github.com/Jayluci4/news) - automatic news pull from 10+ sources and generate ai summaries
- [**india-grants-oracle**](https://github.com/Jayluci4/india-grants-oracle) - Oracle system for discovering government grants

---

## 💡 How to Learn AI from Scratch: Complete Roadmap

### Step 1: Master the Fundamentals (Start Here)

**Prerequisites:** Basic Python, linear algebra, and calculus

**Recommended starting point:**
- [Karpathy's micrograd](https://github.com/karpathy/micrograd) - Learn backpropagation and gradient descent from scratch (~150 lines)
- Understand: Neural networks, forward pass, backward pass, chain rule, optimizers

### Step 2: Sequential Models (This Series - Part 1)

After micrograd, learn how neural networks process sequences:
1. **[micro-rnn](https://github.com/Jayluci4/micro-rnn)** - RNNs, BPTT, vanishing gradients
2. **[micro-lstm](https://github.com/Jayluci4/micro-lstm)** - Gates, cell state, memory
3. **[micro-gru](https://github.com/Jayluci4/micro-gru)** - Simplified LSTM
4. **[micro-embedding](https://github.com/Jayluci4/micro-embedding)** - Word2Vec, semantic vectors
5. **[micro-seq2seq](https://github.com/Jayluci4/micro-seq2seq)** - Encoder-decoder, translation

### Step 3: Modern AI (This Series - Part 2)

Build cutting-edge systems:
- **Tokenization, Attention, Transformers** → Understand GPT & BERT
- **RAG, Agents, Knowledge Graphs** → Build intelligent applications
- **RLHF, Diffusion, LoRA** → Fine-tuning and image generation

### Additional Resources

- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Full GPT training loop
- [Karpathy's nanochat](https://github.com/karpathy/nanochat) - Complete chat model
- [Stanford CS231n](http://cs231n.stanford.edu/) - Convolutional neural networks
- [Stanford CS224n](http://web.stanford.edu/class/cs224n/) - NLP with deep learning

### What's New

**Latest Updates (2025):**
- 5 NEW foundational repos bridging micrograd → transformers
- Complete learning path: 14 repos, ~3050 lines of educational code
- All implementations verified with gradient checking

**Coming Soon:**
- Video tutorials explaining each architecture
- Blog series: "How X Actually Works" for each component
- Interactive visualizations and demos
- Community contributions welcome!

---

## 🎓 About Me

**Jayant Lohia** - AI Engineer, Educator, and Open Source Contributor

- **Expertise:** Deep Learning, Neural Networks, NLP, Knowledge Graphs, Autonomous AI Agents
- **Approach:** First principles thinking - implement algorithms from scratch to build deep intuition
- **Philosophy:** Simplicity beats complexity. Understanding beats memorization. Building beats watching.
- **Teaching Style:** Heavy comments, clear visualizations, minimal dependencies, maximum clarity

**Technical Skills:** Python, NumPy, Machine Learning, Deep Learning, Transformers, LLMs, RAG Systems, Multi-Agent Systems, Knowledge Representation

**Mission:** Make AI accessible to everyone through clear, minimal, educational implementations

---

## 🤝 Connect & Collaborate

**Looking to collaborate on:**
- AI/ML research and education projects
- Open source contributions to educational tools
- Building accessible AI learning resources
- Neural network implementation tutorials

**Available for:**
- Full-time AI/ML engineering roles
- Technical consulting on AI systems, agents, knowledge graphs
- Speaking engagements on AI education and implementation
- Code reviews and educational content collaboration

**Let's talk:** jayantlohia16@gmail.com

**Find me on:**
- GitHub: [@Jayluci4](https://github.com/Jayluci4) - Star repos you find useful!
- Email: jayantlohia16@gmail.com - Questions, collaborations, opportunities
---

## ⭐ Support This Educational Mission

**Help make AI education more accessible:**

🌟 **Star the repositories** - Shows support and helps others discover them
📢 **Share with your network** - Students, engineers, researchers learning AI/ML
💬 **Open issues** - Ask questions, suggest improvements, report bugs
🔧 **Contribute** - Fix typos, add examples, improve explanations
📝 **Write tutorials** - Blog about your learning journey using these repos
🎥 **Create content** - Make videos, courses, or study guides

**Every contribution helps democratize AI knowledge!**

*Used these repos to learn? Drop a star and share your story!*

---

## ❓ Frequently Asked Questions

### How is this different from online courses?

**Online courses** teach you to use frameworks (PyTorch, TensorFlow). **This series teaches you how those frameworks work internally**. You'll implement backpropagation, attention, and transformers from scratch, giving you deep intuition that no course can provide.

### Do I need a PhD to understand this?

**No!** If you know basic Python, linear algebra (matrix multiplication), and calculus (derivatives), you can follow along. Each implementation includes extensive comments explaining the "why" behind every line.

### Why NumPy instead of PyTorch?

**Learning:** NumPy forces you to implement everything manually - you see exactly how gradients flow, how matrices are shaped, how backpropagation works. PyTorch hides these details.

**Understanding:** After building from NumPy, you'll appreciate PyTorch's abstractions and use them more effectively.

### How long does it take to complete the series?

- **Each repo:** 2-4 hours to read, understand, and experiment
- **Foundational Components (Part 1):** 1-2 weeks
- **Modern Architectures (Part 2):** 2-3 weeks
- **Total:** ~1 month of focused learning to go from RNNs to state-of-the-art transformers

### Can I use this for my research/project?

**Yes!** All code is MIT licensed. Use it for research, teaching, projects, or commercial applications. Attribution appreciated but not required.

### What should I do after completing this series?

1. Build a project using what you learned
2. Read the original research papers (linked above)
3. Contribute improvements back to these repos
4. Share your learning journey to help others

### How can I get help if I'm stuck?

1. **Read the comments** - Every implementation is heavily documented
2. **Check the papers** - Foundational research papers section has all references
3. **Open an issue** - Ask questions on the specific repo
4. **Email me** - jayantlohia16@gmail.com for complex questions

---

## 📝 Recent Activity

<!--START_SECTION:activity-->
<!--END_SECTION:activity-->

---

<div align="center">

**"The best way to understand AI is to build it yourself."**

*Building in public. Learning in public. Teaching in public.*

---

[![Follow on GitHub](https://img.shields.io/github/followers/Jayluci4?label=Follow&style=social)](https://github.com/Jayluci4)

</div>
