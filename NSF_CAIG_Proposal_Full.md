# CAIG: GeoAI Traffic Twin to Study the Impact of Sub-Autonomous, Autonomous, and Connected Vehicles in Midwest Corridors

**NSF Futurew CoRE 2026–27 Proposal**

---

## Project Summary

### Overview

This CAIG GeoAI project will build a geospatially faithful, AI-driven digital twin of Indianapolis and surrounding rural counties to study how partial deployment of connected and autonomous vehicles (CAV) interacts with winter weather and heterogeneous infrastructure to reshape traffic patterns and driver behavior. Leveraging the partnerships with the Indiana Geographic Information Office (IGIO) and the Indiana Department of Transportation (INDOT), the project embeds multi-lane geometry, control devices, land use, elevation, and snow/precipitation fields directly into TransModeler, creating a realistic testbed that goes far beyond abstract graph-based AV simulations.

Scientifically, the work addresses four major gaps: (i) the lack of GIS-grounded AV simulation for snow-affected, mid-sized regions; (ii) over-reliance on steady-state metrics (average speed, delay) instead of emergent phenomena; (iii) limited treatment of winter weather and rural network sparsity; and (iv) weak integration of complexity theory with spatial network science in the context of mixed autonomy. Three integrated research thrusts tackle these issues.

**Thrust I** constructs and validates the digital twin using ATLAS data, LiDAR-derived sight distance and grade, INDOT traffic counts, and historical weather, producing a lane-level baseline of 100% human-driven traffic for Indianapolis and adjacent counties.

**Thrust II** develops multi-agent reinforcement learning and game-theoretic controllers for mixed human, semi-autonomous, and fully autonomous vehicles, training them in the GIS-based twin under varied demand, weather, and incident scenarios to evaluate safety, efficiency, equity, and robustness.

**Thrust III** applies complexity and network science—phase transitions, percolation thresholds, cascade dynamics, and synchronization metrics—to identify critical CAV penetration levels and vulnerable corridors where small changes in automation or weather trigger system-wide congestion or resilience gains.

### Intellectual Merit

The intellectual merit lies in uniting high-fidelity GIS infrastructure, multi-agent AI, and complexity metrics to derive new theory about mixed human–AV systems while also creating scalable GeoAI methods. Primary innovation is a CAV-ready digital twin, which includes algorithms for CAV controls, with a goal to achieve realistic simulation of mixed traffic. Methodological novelties include multi-agent reinforcement learning in realistic spatial networks, physics-informed driver models, and complexity-aware control strategies that anticipate phase transitions instead of merely reacting to them. The project will use novel economics and algorithmic game theory for understanding how human driving behavior changes with varying degrees of autonomous vehicle adoption. The project is inherently multidisciplinary, borrowing reinforcement learning from AI, network games from economics and game theory, simulation and modeling from statistics, modern AV transportation networks from transportation engineering, and GIS backbone from Geography.

### Broader Impacts

Broader impacts center on open, reproducible infrastructure and practitioner capacity-building. The team will release the TransModeler–RL interface, large anonymized simulation and complexity datasets, and a percolation/cascade analysis toolkit under open licenses, while delivering a CAV policy evaluator dashboard, resilience maps, and policy briefs for INDOT, the City of Indianapolis, and other agencies via IndianaMap and Indiana University's DataCORE repository. The project will support 2 PhD students who will be experts on transportation systems and AI. Education materials from this project will advance AI education coupled with GIS technologies. Through annual workshops, this project will help mid-sized and rural regions plan for autonomous vehicles using scientifically grounded, geospatially explicit evidence rather than extrapolations from coastal AV pilots.

---

## 1. Overview

### 1.1 Introduction

The research objective of this proposal is to develop a digital twin for city traffic simulation to study the impact of sub-autonomous, autonomous, and connected (SAC) vehicles on the existing traffic network systems in central Indiana, a large metropolitan in a Midwestern state of the USA. Specifically, we will study the impact of SAC vehicles on overall traffic patterns, human driver behaviors, and traffic incident management under varying weather and varying degrees of SAC penetration.

The above objectives will be accomplished by developing a collection of simulation models on **TransModeler**, a traffic simulation and transportation modeling software capable of simulating traffic flow on real-life road networks for analyzing congestion, travel times, and traffic control strategies at the micro-, meso-, and macro-levels, providing real-world behavior of traffic. By leveraging our existing research collaboration with INDOT (Indiana Department of Transportation), we will use real traffic data to calibrate the digital twin parameters by training Gaussian Process (GP)-based surrogate models. Besides, in these simulation models, we will supersede the existing driver-centric vehicle dynamics (car-following, lane-changing, merging) of TransModeler by multi-agent dynamics where the agents are trained by machine learning models to maximize a mixture of competitive and collaborative objectives.

Successful completion of this research will produce a virtual twin of Central Indiana city traffic, capable of accurately answering the impact of SAC vehicles on traffic pattern, driver behavior, and traffic incident management. It will also develop novel artificial intelligence (AI) technologies for training agent drivers, who are capable of mimicking human and SAC drivers, by borrowing methodologies from machine learning, game theory, network science, and complexity theory.

Our digital twin will be built entirely within the TransModeler traffic simulation platform. By using ATLAS maps road model, we will use TransModeler's programming API on top of its native Geographic Information System (GIS) map layers—roads, ramps, intersections, signals, stop signs, and speed limit—to construct simulations that operate directly on a geospatially faithful digital twin of Indianapolis and surrounding rural counties. The research will leverage expertise and collaboration built through an existing Indiana State ATLAS MAPS grant awarded to this research team for validating an 11-county Network GIS model for Indiana.

Such GIS-first architecture allows AI agents (trained with multi-agent machine learning and game-theoretic controllers) to perceive and act in true geographic space, so that learned strategies, detected phase transitions, and policy experiments are immediately interpretable, auditable, and comparable to alternative transportation policies in the exact same spatial frame that planners and geoscientists already use. Without a GIS backbone, network and complexity science risk devolving into "toy" platforms, where the mathematics of phase transitions and percolation is sound but experimental validation is excruciatingly naïve and impossible to translate for agencies.

GIS provides the most accurate, validated representation of transportation infrastructure, enabling lane-level topology, control devices, land use, elevation, and weather fields to coexist in a single spatial database and to be interrogated as a GeoScience object rather than as a stylized graph. The usage of GIS as the core integrative layer makes this CAIG project genuinely co-equal in AI and Geosciences, rather than an abstract AI exercise on a toy network.

### 1.2 Vision, Goals, and Approach

With rapid development in sensing and communication technologies, together with algorithms and computing capabilities, tremendous efforts are being devoted to the research, development, and testing of sub-autonomous, autonomous, and connected (SAC) vehicles. Several studies project that a significant portion (nearly 10% to 30%) of vehicle fleets will be Level 4 automated vehicles (AV) in the 2030s. Companies like Waymo (Alphabet) have already started commercial Robotaxi services in cities like San Francisco, Austin, LA, Miami, and Phoenix. Tesla recently launched limited Robotaxi service in Austin.

However, AV pilots in San Francisco, Austin, and Phoenix operate in temperate climates, dense urban environments, and absence of heavy snow. These studies illuminate high-penetration autonomy (40–100%) but leave critical questions unanswered for mid-sized, snow-affected regions like Indianapolis that are not pilot sites and will encounter mixed human–AV traffic under challenging seasonal conditions. This research will investigate the impact of AV for such regions by considering winter driver conditions, rural infrastructure sparsity, and realistic driver adaptation at partial AV penetration.

In existing literature, most autonomous vehicle (AV) simulation studies model road networks as abstract graphs composed of nodes and links, emphasizing topological connectivity and traffic flow dynamics while largely abstracting away detailed geospatial structure. Widely used microscopic and agent-based traffic simulators, such as **SUMO** and **MATSim**, represent road infrastructure primarily through graph-based networks augmented with limited geometric attributes, rather than full-fledged GIS semantics. Simplified network representations in analytic and simulation-based approaches are easier to study capacity, stability, and control, but they come at the expense of spatial fidelity. Specifically, graph abstractions alone are insufficient to capture lane-level geometry, curvature, and spatial context, requiring the integration of detailed geospatial data for high degree of realism, which is the primary motivation for this work.

Another grand motivation of this work is to incorporate network theory and economics theories for studying realistic traffic networks and user behavior. While existing traffic simulation works focus on understanding steady-state metrics (average speed, delay) rather than emergent phenomena such as phase transitions, percolation thresholds, and cascade failures, there is limited integration of complexity theory and spatial network science to explain system-level reorganization under mixed autonomy. In fact, no prior work combines GIS-embedded infrastructure, multi-agent AI, and complexity metrics to study how network topology, snow/weather, and rural–urban structure jointly shape emergent traffic behavior under partial autonomy.

**TransModeler** is often chosen for large regional planning and network studies, particularly in the USA, and **Aimsun** has a larger footprint in Europe. While both software are capable of multi-resolution and hybrid traffic simulation through API and scripting, TransModeler has much stronger GIS integration and is better for large operational performance and planning. For our research, we will consider both platforms, but we are more inclined towards TransModeler for multiple reasons: (1) Co-PI is an expert on TransModeler with many years of experience; (2) PI and Co-PI are currently working with a research project with INDOT using TransModeler for validating network GIS models for Indiana; (3) TransModeler is best for simulating large networks for dynamic traffic assignment.

### 1.3 Research Tasks

Our overall research objectives can be decomposed into three different research thrusts:

#### Thrust I: Construct a Geospatially Faithful TransModeler-Based Digital Twin of Indianapolis and Rural Hinterlands Transportation Network to Investigate the Impact of CAV Penetration

In this thrust, we will build a lane-level GIS network by integrating ATLAS maps street centerlines, USGS (United States Geological Survey) LiDAR-derived elevation/sight-distance, functional road classifications, INDOT (Indiana Department of Transportation) traffic counts, and 10-year snow/precipitation records. The network will span interstates (I-65, I-70), arterials, collectors, and rural connectors in Marion County and surrounding counties (Hendricks, Hamilton, Johnson, Boone, Morgan). This network will serve as the geospatial foundation for all simulation experiments.

The major research task of this thrust is to study the impact of varying degrees of CAV penetration on the above road networks. To facilitate intelligent transportation systems, we will consider street lights or other digital signs as roadside units (RSUs). Three kinds of communication protocols will be utilized: Vehicle-to-Vehicle (V2V) for communication between two autonomous vehicles in close proximity, and Vehicle-to-Infrastructure/Infrastructure-to-Vehicle (V2I/I2V) to facilitate communication between vehicles and RSUs. Then, considering varying degrees of CAV penetration, we will develop models and algorithms for: (1) Route Guidance, (2) Freeway Speed Harmonization, and (3) Traffic Light Optimal Speed Advisory. For each of these algorithms, their performance and limitations will be analyzed for different weather conditions.

#### Thrust II: Research on Human-AV Driver Behavior by Developing and Integrating Multi-Agent AI Models

In this research thrust, we will implement three vehicle classes in TransModeler: (i) human-driven vehicles calibrated from field data and behavioral literature, (ii) partially automated vehicles (SAE Levels 2–3) with semi-autonomous decision rules, and (iii) highly automated/connected vehicles (SAE Levels 4–5) with AI-based multi-agent reinforcement learning (MARL) and cooperative game-theoretic controllers. The human drivers will follow calibrated car-following (e.g., Intelligent Driver Model), lane-changing (e.g., MOBIL), and gap-acceptance models. AV agents will use deep reinforcement learning (DRL) to optimize individual and collective objectives (minimizing travel time, maximizing safety, reducing emissions) while adapting to human driver behavior and weather-induced hazards.

Specifically, we will validate whether **multi-objective MARL can enable AVs to operate safely and efficiently alongside human drivers across a range of automation penetration levels, and cooperative game-theoretic mechanisms (e.g., signaling, negotiation) will outperform selfish optimization in reducing system-wide congestion and improving safety.**

#### Thrust III: Research on CAV Impact in Terms of Emergent Complexity Phenomena and Develop Complexity-Based Metrics for Network Resilience

In this research thrust, we will use tools from complexity theory and spatial network science to identify critical thresholds in the simulated systems. Specifically, for a mixed traffic environment, we will quantify phase transitions (sudden shifts in congestion regimes), percolation thresholds (minimum AV connectivity needed to maintain flow in snow events), and cascade onset points (where small perturbations trigger system-wide delays). We will map these metrics spatially to identify corridors and intersections most sensitive to AV penetration and weather. We will develop a complexity-theoretic framework that allows planners to predict when marginal increases in AV share will produce large system-level changes.

This research will investigate whether **complexity metrics—particularly percolation thresholds and synchronization breakdowns—can be mapped and estimated from network topology and traffic demand patterns, enabling agencies to proactively design infrastructure and policies that either stabilize transitions or exploit them for congestion reduction.**

### 1.4 System Architecture

Central to the system is the TransModeler simulation engine. It will ingest various data: ATLAS maps, LiDAR (from USGS), weather (from NCEI), and traffic, both real-time and historic (from INDOT). We will also develop different algorithms using TransModeler API/GISDK (Geographic Information Systems Developer's Kit), which the vehicle agents of the TransModeler simulation engine will use. Multi-agent AI and game-theoretic-based learning modules will be developed and trained outside TransModeler, and they will be optimized by data streams obtained from the TransModeler simulation engine. From census data, a collection of synthetic travel demand will be generated, and then using our developed algorithms, vehicles will run in the simulation engine to meet those traffic demands. Simulation parameters will be learned by using INDOT traffic data and Gaussian Process (GP) as a black-box optimization method. The overall deliverables of this work will be a digital twin of the Indiana traffic network within TransModeler along with a collection of ML and AI models for executing various algorithms that the vehicle agents in TransModeler execute.

### 1.5 Intellectual Merits

#### Intellectual Merit 1: Novel Integration of AI Methodologies with GIS-Platformed Traffic Simulation

The most important intellectual merit of this project is the innovative integration of artificial intelligence, network theory, and economic models into traffic simulation study to understand AV impacts on traffic infrastructure. Novel integration of MARL (multi-agent reinforcement learning) with spatial network models and traffic simulation proposed in this work enables agents to learn from realistic, geographically constrained environments. Besides, this project will integrate winter weather, snow accumulation, and visibility reduction—phenomena underrepresented in existing AV simulation literature.

Typical MARL applications use abstract, homogeneous environments or small-scale simulations. This research will operationalize MARL in a realistic, heterogeneous GIS-embedded traffic network with thousands of agents, non-stationary environments (weather, traffic demand), and complex constraints (intersection topology, lane geometry, traffic signals). Novel algorithms will be developed and integrated into the simulation engine through TransModeler's API/scripting (GISDK/Caliper Script). These efforts constitute transformative research as they will open a new research direction where machine learning is used in a novel way to study the impact of autonomous vehicle adoption in real-life traffic networks.

This grant directly advances Yann LeCun's vision of physical AI—AI systems that understand and interact with the physical world through embodied experience—by creating a geospatially grounded digital twin where autonomous vehicle agents learn to navigate real-world constraints: winter weather physics (friction coefficients, visibility reduction), infrastructure geometry (sight distances from LiDAR), and emergent social dynamics (human-AV interactions in mixed traffic).

#### Intellectual Merit 2: Novel AI Models for Understanding Human Driving Behavior

Another intellectual merit of this research is to use economics and algorithmic game-theoretic models to understand driver behavior with significant AV penetration in transportation networks. The geosciences community faces critical unsolved questions at the intersection of transportation geography, human behavior modeling, and spatial adaptation to environmental changes. It is not yet studied how driver behavior varies under different automation levels across diverse infrastructure and weather conditions. Current transportation models treat drivers as rational agents with static parameters, ignoring adaptive behavior, risk perception, and environmental stress.

This research will seek answers to these unanswered questions: How will behavior change with significant autonomous vehicle adoption along with the availability of massive data from V2V and V2X communication? How can we leverage cooperative game theory to encode social objectives (safety, fairness) into agent behavior, advancing beyond purely selfish optimization? Besides, mixed human-AV traffic would create spatial conflict zones where human and autonomous driving assumptions collide.

#### Intellectual Merit 3: Application of Complexity Theory to Understand AV Impact on Network Resilience

A key intellectual merit of this work is its utilization of complexity theory to study network resilience of mixed-autonomy traffic systems embedded in realistic GIS infrastructure. This effort will produce practitioner-ready knowledge—spatially explicit vulnerability maps and decision thresholds that can be used by transportation agencies for infrastructure investment and policy design.

By using emergence theory and percolation analysis to design RL reward structures that incentivize AVs to operate near criticality—where small individual actions can have large system-level effects—the project will provide system-level resilience and adaptability. The proposed complexity-aware AI methodologies can model and predict spatial emergence of congestion, safety failures, and optimization zones under different automation scenarios, generating predictive maps of vulnerability and adaptation capacity.

### 1.6 Team Composition and Expertise

The project is led by two experts from two disciplines with complementary expertise.

**PI Dr. Mohammad Hasan**, Professor of Computer Science, is an expert in machine learning, AI, and network theory. He has substantial experience in developing machine learning models for various network-related problems: including network alignment, network sampling and counting, network embedding models based on deep learning, link prediction, and ordered embedding in directed networks. He has also proposed methods for extracting entities from sentences for building a knowledge map, and is currently leading the construction of a knowledge map in the Computer Science discipline for public use.

**Co-PI Dr. Aniruddha Banerjee** serves as the Principal Investigator for Indiana's ATLAS MAPS, the 11-county GIS network infrastructure foundational to this proposal. He bridges theoretical network science and operational systems as the holder of U.S. Patent No. 11,853,953 B2, which utilizes topological reduction (Laplacian transforms) for autonomous aerial vehicle networks. This expertise culminated in the development of SkyDOS, a drone operating system built upon his 20-year history of customizing TransModeler, the industry's leading GIS-integrated traffic simulation software. Dr. Banerjee's research further integrates network complexity and spatial analysis with transportation systems, specializing in network fragmentation and Bayesian hierarchical modeling.

The PI and Co-PI will collaboratively pursue the three thrusts of the project, with Dr. Hasan focusing on all three and Dr. Banerjee on Thrusts II and III. Project evaluation will be conducted by **Dr. Steven Aldrich** and **Ms. Marianne Cardwell**, independent experts from the Indiana Geographic Information Office.

---

## 2. Thrust I: Construct a TransModeler-Based Digital Twin of Central Indiana Transportation Network to Investigate the Impact of CAV Penetration

### 2.1 Overview

Digital twin (DT) is a mapping of real-world physical entities into a simulation environment using data from the physical world to accurately reflect the lifecycle process of corresponding physical entities. In this research thrust, our first task is to build a digital twin of the transportation network (DT-TN) of Central Indiana in TransModeler. To achieve this goal, we will first integrate ATLAS maps street centerlines, USGS LiDAR-derived elevation/sight-distance, and functional road classifications of Central Indiana with the GIS backbone of TransModeler. Current and historical weather data will also be ingested to make weather a model parameter in all the algorithms that will be developed.

To make a realistic DT of the transportation network, we need to minimize the reality gap between the physical system and the DT by calibrating the learnable parameters of DT using data from real life. To accomplish this task, we will leverage our existing collaboration with INDOT to obtain real-time and historic traffic data. The initial version of DT-TN will have no CAV penetration; hence, once calibrated, this version will be able to simulate the existing Central Indiana transportation network faithfully.

The major research task then will be to incorporate CAVs in the simulation by leveraging TransModeler's API support. In this environment, traffic lights and other digital roadside signs become roadside units (RSUs); AVs (and possibly human-driven cars) in close proximity communicate using V2V networks; and AVs and RSUs communicate using V2I/I2V. For all such communications, DSRC (Dedicated Short-Range Communication)-based wireless technology is used. Besides using TransModeler's native support for AV, we will also extend the AV infrastructure in TransModeler to fully emulate the mixed traffic scenario.

Once the CAV simulation environment is built, we will develop models and algorithms for:

1. **Route Guidance**
2. **Freeway Speed Harmonization**
3. **Traffic Light Optimal Speed Advisory**

### 2.2 Related Research

With the rapid development of SAC vehicle technologies, there has been growing interest in developing systems for traffic simulation. Such techniques are broadly categorized into rule-based and learning-based approaches. Rule-based methods employ analytical models, such as the Intelligent Driver Model and its derivatives. Such methods are simple and scalable as they simulate individual vehicles' longitudinal dynamics and reproduce real congested traffic patterns (free flow, stop-and-go waves) by modeling interactions via speed, gap, and relative velocity. Such methods are better suited for building digital twins once the parameters are well-calibrated with real-world data.

On the other hand, learning-based models utilize deep generative models trained on trajectory datasets to mimic real-world driving behaviors. They are data-driven and are able to handle human-like interaction and multi-agent behavior. Such methods can also exploit generative models to create synthetic data depicting complex scenarios to study rare and specific traffic events.

Unfortunately, learning-based approaches are often trained on specific datasets and are designed to solve a specific task, so they are not a good fit for understanding the general impact of CAV on an existing traffic network.

Our proposed research is different from the above works in several ways:

1. We will develop a digital twin (DT) of a Midwestern state transportation network by feeding real-time traffic, weather, and geographical data to faithfully capture the existing traffic network in a simulated environment. Substantial efforts will be used to ensure the exact reproduction of existing traffic networks in the DT's simulated environment, by properly training surrogate models.

2. We will be working closely with INDOT to ensure that our DT produces analytics and results that are useful not only for research on traffic engineering but also for developing future policies for transportation system management.

3. From the methodological aspect, instead of using pure rule-based or learning-based approaches, we will adopt a mixed strategy. For vehicle dynamics, we will extend traditional rule-based approaches (e.g., car-following model) with CAV adaptation of those models. We will also develop algorithms that take advantage of connected vehicle communication protocols, utilizing methodologies from learning-based methods, such as reinforcement learning and deep neural networks.

### 2.3 Research Plan

Building a well-calibrated digital twin of the transportation network (DT-TN) in TransModeler is the primary research task for this thrust. TransModeler provides vehicle-following, acceleration, deceleration, lane-changing, merging, and yielding behavior, tunable based on driver aggressiveness and road geometry. For autonomous and connected vehicle simulation, TransModeler also provides V2V and V2I/I2V communication and sensing APIs for developing custom vehicle control logic, such as cooperative driving or connected adaptive cruise control (CACC) strategies, which we will use to control vehicle dynamics instead of using TransModeler's default vehicle control module.

For calibrating DT-TN, we will use **Gaussian Process (GP)** , a probabilistic surrogate model capable of emulating complex, computationally expensive physical simulations. In recent years, GP has been increasingly used as a surrogate model to build digital twins in various domains, such as healthcare and manufacturing applications, due to its ability to calibrate simulation parameters accurately with fewer data points.

We will calibrate parameters to reproduce the following behaviors in the DT across different scales:

- **Microscopic**: speed, lane change, trajectory smoothness
- **Mesoscopic**: travel time, queue lengths, throughput
- **Macroscopic**: flow-density relation, congestion clusters, network-wide vehicle miles traveled

Major calibratable parameters will include vehicle dynamics control parameters (desired speed, time headway, max acceleration, reaction delay), human behavior parameters (lane-changing aggressiveness, driving above speed limit propensity), environmental parameters (capacity drop factors, incidents and weather multipliers), and AV-related parameters (communication latency, communication failure rate, etc.). Most parameters will have distributions based on space, time, and context such as weather and traffic state.

#### Route Guidance

For AVs, route guidance is important to manage congestion caused by rush hour traffic and traffic incidents. In the case of a traffic incident, the incident link ID will be captured by roadside units (traffic light cameras and sensors) or reported by AVs using V2I messages. The active RSU will then relay the message to all AVs planning to take the link associated with the incident.

In this game, vehicles are players and routes are strategies which players choose to minimize their payoff/cost. Commuting is habitual, so we will repeat these games, allowing the players to learn and adapt over time using reinforcement learning. Purely selfish behavior in such games leads to "Nash Flow," where no driver can improve their utility by changing routes. However, Nash Flow is not socially optimal. Repeated congestion games' objective is to nudge the Nash equilibrium closer to the social optimum. We will use **mean-field Reinforcement Learning** by allowing AVs to learn routing policies through interaction with the network and the environment.

#### Freeway Speed Harmonization

Freeway speed harmonization in CAV is an active traffic management strategy that uses real-time data to adjust vehicle speed by leveraging V2I communication, thereby smoothing traffic flow and increasing throughput. In our algorithm, we will consider traffic, weather, and lane merging bottlenecks, as well as mixed traffic scenarios where CAVs coexist with human-driven vehicles.

In our implementation, we will follow Wei et al.'s reinforcement learning framework, which uses a reward function considering smooth speed change, reduced oscillation, minimal energy consumption, safety, and driver comfort. All CAV vehicles will iteratively explore and update their policies using policy gradient and actor-critic approaches to find optimal speeds to maximize their reward function.

#### Traffic Light Optimal Speed Advisory

In our DT-TN, RSUs will get real-time signal information from the connected intersection signal controller and broadcast this information to the CAVs within their communication range. Based on this, they will adjust their approaching speeds according to the optimal speed advisor algorithm. Our main objective is to study whether intersection performance (throughput and safety) can be improved if CAVs can dynamically adjust their speeds according to real-time traffic signal information.

We will not only control drivers' approaching speed but also optimize signal timings and phases based on data from approaching CAVs. In existing literature, there already exists a collection of algorithms for this task, often referred to as **GLOSA (Green Light Optimal Speed Advisory)** methods, but most of these methods are not evaluated under real traffic data in a digital twin environment. We will compare these methods and validate their effectiveness under various degrees of CAV penetration and varying weather scenarios.

Once the above algorithms are implemented and trained, we will run extensive simulations of everyday traffic of Central Indiana using digital twins, under different CAV penetration, varying weather, and traffic conditions. The impact of CAV on the traffic network will then be measured by comparing multiple factors, including travel time and delay, user cost, traffic flow and density, environmental impact, accident rate, and overall benefit-cost analysis.

### 2.4 Deliverables

Deliverables of this Thrust include:

- A software implementation of the Central Indiana traffic network digital twin (DT) inside the TransModeler platform, in which connected and autonomous vehicle features are fully realized.
- Calibration algorithms to fix parameters of the digital twin so that the simulation environment behaves very similarly to the real-life traffic data injected into the DT.
- Implementation of CAV vehicle control algorithms so that by generating CAV routes using these algorithms, we can study and analyze the impact of CAV penetration in Central Indiana traffic.

---

## 3. Thrust II: Research on Human–AV Driver Behavior by Developing and Integrating Multi-Agent AI Models

### 3.1 Overview

In this thrust, we will develop and validate multi-agent artificial intelligence models to capture the complex behavioral dynamics of mixed-autonomy transportation networks, where human-driven vehicles (HDVs), partially automated vehicles (SAE Levels 2–3), and highly automated/connected vehicles (CAVs, SAE Levels 4–5) interact under varying weather and traffic conditions.

We will test whether AVs equipped with multi-objective MARL and cooperative game theory can safely and efficiently coexist with human drivers, and whether cooperative strategies outperform selfish optimization in reducing congestion and improving safety. For this research, we will introduce three vehicle classes into the TransModeler digital twin:

1. **HDVs** with calibrated car-following using IDM (Intelligent Driver Model), lane-changing using MOBIL (minimizing overall braking induced by lane-change), and gap-acceptance models validated against Indiana field data.
2. **Partially automated vehicles** with deterministic semi-autonomous rules representing adaptive cruise control and lane-keeping assist.
3. **CAVs** controlled by deep reinforcement learning agents optimizing multi-objective rewards (travel time, safety, emissions, coordination) while adapting to human behavior and weather-induced hazards.

The core innovation is to combine individual vehicle intelligence (single-agent DRL for tactical maneuvers) with system-level cooperation (multi-agent coordination via signaling, negotiation, and implicit communication) to achieve emergent traffic flow optimization without centralized control.

### 3.2 Related Work

Related research for Thrust II spans four main areas:

1. **Multi-agent reinforcement learning (MARL) for autonomous driving**: Zhang et al. survey MARL for autonomous driving, covering environment design, observation and action spaces, interaction modeling, and algorithmic choices for cooperative and competitive settings. Hegde et al. review MARL methods for safe lane changes by CAVs, emphasizing reward design, safety constraints, and interaction-aware policies.

2. **Mixed-motivation and social MARL**: This literature motivates our use of multi-objective reward structures in which cooperation weights depend on local traffic density, allowing pursuit of socially efficient equilibria rather than purely individualistic Nash equilibria.

3. **Cooperative and game-theoretic control for CAVs**: Approaches such as coalitional and cooperative games for joint strategy formation and Stackelberg games for merging have demonstrated improvements in safety, efficiency, and environmental performance. These results motivate our research using Stackelberg-based ramp-merging controllers and Nash-bargaining-based intersection managers.

4. **Human driver behavior and HDV–AV interaction modeling**: Human driver behavior models, including IDM and MOBIL, have been extensively studied and calibrated on highway and arterial data. This work underpins our construction of a stochastic environment in which MARL CAVs must learn to interact with heterogeneous human drivers.

### 3.3 Research Plan

Our research plan comprises four phases:

#### Phase 1: Calibrate Human Driver Models

We will calibrate human driver models using loop detector data from interstate corridors around Indianapolis, INDOT SmartLane video trajectories, and NOAA weather observations to estimate IDM, MOBIL, and gap-acceptance parameters under clear, light-snow, and heavy-snow conditions. Calibration proceeds from aggregate speed–flow curves to trajectory-level parameter estimation and clustering into driver archetypes, followed by validation against observed speed distributions, lane-change frequencies, and time-headway distributions. The deliverables are nine parameter configurations (three driver types by three weather regimes) and a documented calibration report.

#### Phase 2: Train MARL-Based CAV Agents

We will train MARL-based CAV agents in a TransModeler–OpenAI Gym environment exposing reset/step APIs, local observations (ego state, nearest vehicles, road geometry, weather, V2V messages), and a continuous action space for acceleration and lane-change commands. Training scenarios will systematically vary CAV penetration, demand, weather, incident presence, and network segment to train a **Multi-Agent Proximal Policy Optimization (MAPPO)** with a shared actor and centralized critic over tens of millions of steps using curriculum-based adversarial learning, entropy regularization, and ablation baselines (selfish MARL, independent DRL, rule-based CAVs).

#### Phase 3: Game-Theoretic Coordination Modules

We will implement game-theoretic coordination modules in ramp-merging zones and unsignalized intersections, using **Stackelberg-style controllers** in merges and **Nash bargaining** to select fair, collision-free crossing orders at intersections. Hybrid controllers will combine MARL for nominal driving with explicit game-theoretic logic in critical zones. Comparative experiments will evaluate pure MARL and hybrid configurations on delay, throughput, near-miss frequency, fairness, and worst-case delay.

#### Phase 4: Comprehensive Simulation and Behavioral Analysis

We will execute a comprehensive simulation matrix over CAV penetration levels, semi-autonomous shares, HDV heterogeneity modes, demand and weather conditions, incident/no-incident cases, and multiple network layouts (e.g., I-65 corridor, I-70/I-65 interchange, SR-37), resulting in hundreds of scenarios and thousands of simulation runs.

Behavioral analysis and interpretability work will:

- Characterize emergent behaviors (platooning, implicit communication, adaptation to HDV types)
- Build a taxonomy of human–AV interaction patterns (cooperative yield, assertive overtake, hesitant merge, simultaneous lane change)
- Apply Hidden Markov models and clustering to detect latent states
- Use Shapley value analysis to identify which inputs most influence policy decisions
- Conduct failure mode analysis to classify all collisions and evaluate counterfactual mitigations

### 3.4 Deliverables

This thrust will generate:

- At least **2 journal articles and 3–4 conference papers** on MARL for mixed-autonomy traffic, game-theoretic CAV coordination, human–AV interaction modeling, and winter-weather robustness.
- **Software components**: a TransModeler–RL Gym interface, calibration scripts, and selected trained policies released under an open-source license.
- The **CAIG-MixedTraffic dataset** of trajectories and summary metrics.
- A web-based **CAV Policy Evaluator dashboard** that visualizes performance and safety outcomes across automation mixes, weather conditions, and network segments.

---

## 4. Thrust III: Emergent Complexity in Mixed Human–CAV Networks

### 4.1 Overview

In Thrust III, we will investigate how connected and autonomous vehicles (CAVs)—including sub-autonomous (SAE Levels 1–3), autonomous (SAE Levels 4–5), and connected vehicles—induce emergent complexity phenomena in large-scale transportation networks by interacting with human drivers, network topology, and weather. By integrating outputs from the TransModeler-based digital twin developed in Thrust I and the multi-agent AI models from Thrust II, we will identify critical CAV penetration thresholds at which marginal changes in adoption trigger phase transitions, percolation failures, and cascading congestion.

Phase transitions in a dynamic system occur when system-level properties change discontinuously despite continuous variation in a control parameter, such as density or CAV adoption. For traffic systems, phase transitions can cause new connectivity emergence, and the study of this phenomenon is called **percolation**.

**Central Research Questions:**

1. **Phase transition detection**: How can we quantify sudden shifts in congestion regimes (free flow → synchronized flow → jammed) as functions of CAV penetration, demand, and weather?
2. **Percolation thresholds**: What minimum CAV connectivity or penetration rate is required to sustain network-wide flow under adverse conditions such as heavy snow and reduced visibility?
3. **Cascade onset prediction**: At which spatial locations and CAV penetration levels do local perturbations (incidents, signal failures, cyber or communication disruptions) trigger system-wide delays?
4. **Predictive framework**: Can complexity metrics derived from network topology, baseline demand, and CAV control policies predict these critical thresholds *ex ante*, enabling proactive investment, incident management, and CAV deployment strategies for Indiana corridors?

### 4.2 Related Research

Li et al. is one of the most prominent works on percolation and phase transitions in traffic networks. They show that real city networks undergo percolation-like transitions, where the largest high-speed "functional cluster" of links suddenly fragments as congestion spreads, and the second-largest cluster peaks at the percolation threshold, revealing bottleneck links whose failure disintegrates global connectivity.

Cascading failures and multimodal resilience analysis is a form of network robustness analysis, which shows that large-scale disruptions in transport networks are often driven by cascades that propagate across roads, transit lines, and cyber layers rather than by isolated local failures.

The mixed-autonomy literature shows that appropriately controlled AVs can suppress stop-and-go waves, increase bottleneck throughput, and shift networks from congested to high-throughput phases when penetration reaches moderate levels, often in the 20–40% range. Game-theoretic and learning-based approaches further examine how AV policies interact with human drivers and heterogeneous environments.

These findings motivate Thrust III's emphasis on topology-dependent CAV penetration thresholds, mapping phase transitions in mixed human–CAV networks, and quantifying when CAV coordination suppresses or amplifies congestion cascades.

### 4.3 Research Plan

We will proceed in six tightly integrated phases:

#### Phase 1: Simulation Data Ingestion

Ingest simulation outputs from Thrust I and Thrust II, including vehicle trajectories, link-level speeds and densities, and event logs across scenarios that systematically vary CAV penetration (0–60%), weather (clear vs. light and heavy snow), demand (peak, off-peak, and special events), and perturbations (incidents and cyber or communication failures).

#### Phase 2: Phase Transition Analysis

Compute multiple order parameters—average speed, flow homogeneity, stop-and-go frequency, and network throughput—and estimate critical CAV penetration p\* by examining derivatives and higher-order sensitivities with respect to p_CAV, using bootstrapping to assess statistical significance.

We will use CAV penetration as a control parameter that can drive phase transitions in network-level traffic states. The network-wide traffic state will be represented using order parameters analogous to those in many-body systems, such as the average network velocity φ = (1/N)∑v_i, where φ is the mean segment speed across N road segments, and critical CAV penetration is p\*_CAV, for which the derivative dφ/dp_CAV displays discontinuities (first-order transitions) or diverging sensitivity (second-order transitions).

#### Phase 3: Percolation Analysis

Perform percolation analysis by constructing CAV communication graphs at each time step, computing the largest connected component size as a function of p_CAV, and fitting scaling relationships to estimate percolation thresholds and exponents under different weather and demand regimes.

Percolation theory provides a complementary framework for analyzing connectivity and coordination in CAV-enhanced networks. In our formulation, nodes represent road segments and intersections, and edges capture connectivity enabled by V2V and V2I communication links. The percolation threshold p_c is the minimum CAV density at which a network-spanning cluster of communicating, coordinated vehicles emerges.

#### Phase 4: Cascade Detection and Characterization

Detect and characterize congestion cascades by identifying trigger events in the simulation, tracking the temporal and spatial spread of speed drops and queue formation, and computing cascade size and severity classes. Cascade failures occur when events such as crashes, lane closures, cyber failures, or signal malfunctions reduce capacity on a subset of links and trigger spillback, rerouting, and subsequent overload on adjacent links. We quantify cascade size S as the total vehicle-hours of delay across all affected vehicles and segments.

#### Phase 5: Synchronization Analysis

Evaluate synchronization and desynchronization of traffic flows using order parameters inspired by Kuramoto models, measuring how velocity oscillations across segments become more or less coherent as CAV penetration and control policies vary, and relating abrupt changes in synchronization metrics to observed phase transitions and cascades.

#### Phase 6: Predictive Complexity Framework

Develop a predictive complexity framework that links network-level features (betweenness centrality, clustering coefficient, functional classifications, signal density, geometric properties, and demand characteristics) to critical thresholds (p\*_CAV, p_c, cascade frequency) via interpretable machine learning models. These models will be trained on Indianapolis networks and tested on adjacent counties to assess generalizability.

We will test the hypothesis that intermediate CAV penetrations can be most vulnerable to cascades when human and CAV behaviors are poorly coordinated, while higher penetrations with well-designed controllers dampen cascades by absorbing shocks and rerouting flows more effectively.

### 4.4 Deliverables

Thrust III will produce four main categories of deliverables:

1. **High-impact scientific publications** on percolation, phase transitions, and cascades in mixed human–CAV networks, including manuscripts that formalize critical thresholds in realistic urban networks.

2. **Open-source software and curated datasets**: A Python-based percolation and cascade analysis toolkit that ingests simulation and event data, constructs multi-layer network representations, and implements percolation and cascade algorithms, along with a derived resilience dataset containing percolation curves, cascade statistics, and vulnerability scores for Indiana corridors.

3. **Decision-support products**: Resilience-aware CAV deployment maps that highlight percolation bottlenecks, high-vulnerability links, and candidate resilience corridors, as well as penetration-threshold curves and lookup tables summarizing critical CAV adoption levels under different demand and weather conditions.

4. **Practitioner-oriented guidance materials**: A policy brief on resilience and cascades in mixed-autonomy networks and a technical appendix documenting algorithms, parameter choices, and validation protocols.

---

## 5. Project Evaluation

For each thrust, we will develop evaluation metrics so that the success of that thrust can be quantified.

**Thrust I**: The main objective is to make the digital twin simulation environment as real as possible, and for this we will use the **reality gap** metric to compare simulated measurements with real measurements. A challenge is that for real data, CAV penetration is zero, so the reality gap metric will be evaluated first for zero CAV penetration and will later be extrapolated as CAV penetration increases smoothly. For CAV driving control algorithms (route guidance, freeway speed harmonization, traffic light optimal speed advisory), we will evaluate their effectiveness by traditionally used metrics. Safety metrics (probability of collision, near-miss) will also be used. We will also perform stress testing and evaluate the system's fault tolerance ability.

**Thrust II**: The major products are reinforcement learning models, which we will evaluate by popular RL metrics such as cumulative rewards, average rewards, etc.

**Thrust III**: We will produce percolation thresholds for cascading behavior. We will validate these thresholds by statistical analysis and hypothesis testing over a large number of sampled simulated data.

### External Evaluation

The project evaluation plan also constitutes external evaluation led by the **Indiana Geographic Information Office (IGIO)** to ensure both scientific rigor and practical relevance for Indiana's transportation agencies. IGIO will coordinate yearly evaluation meetings to review progress on the digital twin's GIS fidelity, data standards, and statewide reusability, using their ATLAS/IndianaMap framework and existing data stewardship practices as benchmarks for data quality, completeness, and conformance to state geospatial standards.

These reviews will assess whether the lane-level network, weather layers, traffic demand models, and vulnerability maps are being produced on schedule, whether they are being documented with ISO-compliant metadata, and whether datasets and workflows are ready for broader publication through IndianaMap and the Indiana Data Harvest.

---

## 6. Educational Plan

Two PhD students will work on this project, who will graduate with integrative expertise in GIS, AI, and complexity science—positioning them for careers at federal agencies (USGS, NOAA, FHWA), technology companies, and research institutions. All students will participate in professional development (conference presentations, journal publications, mentorship networks).

PI and Co-PI will apply for the NSF REU program in the theme of "GeoAI for Transportation Planning"; if funded, an additional 40 undergraduate students can be trained in GeoAI research over the duration of the grant.

Two new courses will be built and jointly taught by PI and Co-PI at IU Indianapolis campus:

1. **Complexity in Spatial Transportation Systems**
2. **GeoAI for Transportation Planning**

Expected enrollment in these courses is about 40–50 students from different backgrounds including geosciences, computer science, civil engineering, and public policy. The curriculum will integrate geoscience, AI, and social science perspectives, building a generation of researchers who can work across disciplinary boundaries.

---

## 7. Dissemination Plan

Research results of this project will be disseminated through conference and journal articles, and software products will be released through PI's official GitHub channel. All results will be published openly without any proprietary restrictions to ensure broad access and reproducibility.

- All code, documentation, and de-identified simulation datasets will be released under open licenses (**MIT** for code, **CC0** for data).
- Patent-eligible innovations may be disclosed to the IU Office of Licensing, but licensing strategy will prioritize open-source adoption over commercial exclusivity.
- We will also disseminate research results with our research collaborator INDOT.
- From the second year of the project, we will hold an **annual workshop** for training 20–30 transportation professionals to use our developed digital twin and decision-support tools.
- The PI and Co-PI team will maintain a **project website** providing links to all educational and scholarly research products.

---

## 8. Broader Impacts

### Broader Impact 1: Informed AV Policy by Making Comprehensive AV Simulation Available and Accessible

At present, research on CAV and their impact on transportation systems is largely funded by federal agencies (USDOT, NIST) and AV industries. Federal Government's research priorities focus on setting standards, analyzing safety, and developing new infrastructure. AV industries' research priorities focus on technology development and field testing. However, the organizations that will bear the direct and immediate impact of CAV are state and local agencies, yet they are not well-funded to perform research to understand the impact on local traffic. The proposed project will democratize research on autonomous vehicle impact for policymakers of small states, mid-size cities, and rural regions.

Policy makers using our simulation platform will be able to obtain spatially explicit maps showing which neighborhoods and corridors will benefit or be burdened by different AV deployment strategies, and analyze safety implications for pedestrians, cyclists, and transit users under different AV penetrations and weather conditions.

### Broader Impact 2: GeoAI Workforce Development for Managing Futuristic Intelligent Transportation Networks

Massive CAV penetration will be realized with futuristic intelligent transportation networks, along with novel communication protocols (V2V, V2X), novel ML models for real-time data analysis, and safety-critical embedded systems. An educational objective of this research is to train students on traffic modeling and simulation (SUMO, VISSIM, Aimsun, TransModeler), network optimization, and AI-driven transportation system design. Beyond training graduate students directly on this project, we plan to train more than 50 undergraduate students through the REU program.

### Broader Impact 3: Advancing Educational AI Coupled with GIS Technologies

The proposed project will contribute to the advancement of research at the confluence of artificial intelligence, GIS, traffic simulation and modeling, human-computer interaction, game theory, and network theory. This research will develop new algorithms combining concepts from reinforcement learning, game theory, and human-computer interaction. On a parallel front, network theory will be applied to real-life traffic networks in the context of various degrees of AV penetration, leading research on responsible AI deployment and infrastructure resilience.

---

## 9. Project Management

The PI and Co-PI will collaborate closely on all tasks. PhD students, guided by the PI and Co-PI, will primarily focus on designing, developing, and evaluating the traffic simulation digital twin on the TransModeler platform.

Starting from Year 1:
- One PhD student will work on building the digital twin (DT) in TransModeler (Thrust I)
- The other student will develop multi-agent RL and game-theoretic controller development for CAV (Thrust II)
- Machine learning models will be developed to calibrate the DT using surrogate models

About the middle of Year 2, once the DT is fully calibrated, large-scale simulation will be performed and the tasks on Thrust III will follow. There will be quarterly meetings with INDOT to ensure that simulation scenarios reflect local priorities and constraints.

---

## 10. Results from Prior NSF Support

### PI Hasan

**Current grant**: "IUSE: EDU: Improving Data and AI Literacy of Liberal Arts Students through Project-based Intervention in Liberal Arts Courses" (10/2024–09/2026, Amount: $399,999)

- **Intellectual Merit**: (1) Defining data science and AI literacy for LA curricula; (2) Development of novel knowledge regarding the alignment of AI and data science across different liberal arts courses; (3) Development of novel intervention mechanisms for infusing data science concepts.
- **Broader Impact**: Improving human resources in the data science and AI domain through a pathway that enables students in different LA majors to utilize data with greater effect in their careers.
- **Product**: In progress.

**Completed grant**: "III: Small: Geometric Constraint based Concept Keyword Embedding for Domain-neutral Knowledge Graph Construction" (10/2019–09/2023, Amount: $494,763, IIS-1909916)

- **Intellectual Merit**: Developing link prediction and text embedding methodologies for building domain-neutral knowledge graphs, capturing semantic relations through geometrical constraints within embedding vectors.
- **Broader Impacts**: Expanding embedding methodologies to non-traditional graph data (knowledge graphs, directed graphs, ontologies, taxonomies, causal graphs).
- **Products**: Publications and a knowledge graph for Computer Science.

### Co-PI Banerjee

Co-PI Banerjee does not have active or recently completed NSF grants.

---

*This markdown document was generated from the LaTeX source files in the NSF_Futurew_CoRE_2026_27_Proposal directory.*