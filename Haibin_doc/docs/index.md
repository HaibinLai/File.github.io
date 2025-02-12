<!-- # Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files. -->

# Welcome to SUSTech HPC lab

About this page

本站多数来源于我个人的大学HPC的学习经历。

## HPC：怎么样算的更快？


在当前的计算机科学领域，我们可以大概将CS研究者分为这样两类：AI、System（当然还有很多，比如理论计算机科学，但大部分的研究者的方向如此）。而在System中，有负责存储数据的数据库，负责传输数据的网络（通信、协议）、负责计算机的安全（网络、软件、硬件）、构造计算机的体系结构（RISC-V、GPU）、搭建计算机软件的软件工程与编译器、管理计算机的操作系统...... 以及最后，负责更高效处理数据、更好利用硬件进行计算的并行计算以及高性能计算方向。或者用俗话说，怎么样算的更快？

从具体来说，可以分为这几个问题：
1. **高计算能力**：HPC系统通常由多个处理器（CPU）或图形处理单元（GPU）组成，能够并行处理大量计算任务，极大提高计算速度。
2. **大规模并行处理**：通过并行计算，多个计算任务可以同时进行，显著缩短计算时间。常见的并行方式包括分布式计算和集群计算。
3. **海量存储**：HPC系统配备高速、大容量的存储设备，用于处理和存储巨大的数据集。
4. **低延迟和高带宽**：高速网络连接使得各个计算节点之间能够快速交换数据，减少计算过程中的延迟。

### HPC应用领域
1. **科学研究**：例如气象模拟、天文计算、物理实验、基因组学、药物研发等领域，HPC能够模拟复杂的自然现象，帮助科学家分析大量数据。
2. **工程设计与仿真**：包括汽车、飞机、建筑等领域的计算流体力学（CFD）、结构力学等仿真，HPC帮助设计更高效、更安全的产品。
4. **人工智能与机器学习**：HPC为深度学习、神经网络等AI模型的训练提供强大的计算资源，支持大规模数据的处理与分析。

### 我们的研究对象
- **超级计算机**：专门设计用于进行大规模计算的机器，具备极高的计算能力，如美国的“Frontier”和中国的“神威·太湖之光”。
- **计算集群**：由多个计算节点（通常是服务器）通过高速网络连接组成，协同处理复杂计算任务。
- **云计算平台**：如Amazon Web Services (AWS)、Microsoft Azure等提供的云服务也支持高性能计算，用户可以租用云资源进行计算。

