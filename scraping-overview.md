# Computational Frameworks for Decompression Model Backtesting: Library Analysis, Simulation Methodologies, and Data Acquisition Strategies

## 1. Executive Summary

The validation of novel decompression algorithms requires a rigorous dual-pronged approach: stochastic simulation to identify theoretical divergence points and empirical verification using real-world physiological data. This report evaluates the technical feasibility of backtesting a "Slab Diffusion Model" (based on Hempleman's linear bulk diffusion theory) against the established Bühlmann ZH-L16C algorithm (a perfusion-limited neo-Haldanian model). The analysis focuses on leveraging three specific open-source libraries—AquaBSD/libbuhlmann, libdivecomputer, and Subsurface—to construct a high-throughput computational pipeline.

The investigation establishes that AquaBSD/libbuhlmann provides a robust, isolated calculation engine suitable for generating control data via Monte Carlo simulations. By wrapping its standard input/output interfaces, researchers can generate thousands of stochastic dive profiles to map the "safety envelope" defined by current industry standards. Concurrently, libdivecomputer and Subsurface offer the necessary infrastructure to mine and normalize real-world dive logs. A distinct methodology is proposed to programmatically harvest "orphaned" dive data from GitHub issue trackers and technical forums, utilizing the libdivecomputer parsing API to transform binary device dumps into time-series data suitable for diffusion analysis.

Furthermore, this report synthesizes findings on the fundamental mathematical incompatibilities between the exponential decay of perfusion models and the square-root-of-time kinetics of slab diffusion. These divergences are critical for designing the simulation constraints; the backtesting engine must specifically target profile topologies—such as short, deep bounce dives and multi-level ascents—where these physical models disagree most sharply. The integration of external datasets, including Kaggle and Diveboard repositories, is analyzed as a supplementary validation stream, though the report prioritizes the high-fidelity sample data extractable from raw dive computer dumps.

## 2. Theoretical Divergence in Decompression Modeling

To effectively design a software architecture for backtesting, one must first define the mathematical properties of the models being compared. The software logic must reflect the underlying physics; simply comparing "stops" is insufficient without understanding the kinetic drivers of those stops.

### 2.1 The Bühlmann ZH-L16C Paradigm (Perfusion Limited)

The Bühlmann ZH-L16C algorithm represents the refinement of the Haldanian model, which has dominated decompression theory for over a century. Its central tenet is that inert gas exchange is limited by perfusion—the rate at which blood flows to tissues—rather than the diffusion of gas across cell membranes. Consequently, the body is modeled as a collection of parallel tissue compartments, each characterized by a specific half-time ($T_{1/2}$).

In this framework, the partial pressure of inert gas ($P_t$) in any given compartment follows an exponential uptake and elimination function. The rate of change is proportional to the gradient between the arterial gas pressure ($P_a$) and the current tissue tension. This results in the classic Haldane equation:

$$P_t(t) = P_t(t_0) + [P_a - P_t(t_0)] \cdot (1 - e^{-kt})$$

Where $k$ is the decay constant specific to that compartment ($k = \ln(2) / T_{1/2}$). The ZH-L16C variant defines 16 such compartments with half-times ranging from 4 minutes to 635 minutes, allowing it to model both fast tissues (blood, brain) and slow tissues (joints, bone marrow) simultaneously.

Safety is determined by "M-values," which define the maximum tolerated supersaturation for each compartment at a given ambient pressure. If the gas tension in any of the 16 compartments exceeds its M-value line, the model dictates a decompression stop. This deterministic behavior makes the Bühlmann algorithm an ideal "control" for backtesting: it produces a definitive, binary Pass/Fail result for any given pressure-time profile based on widely accepted safety margins.

### 2.2 The Slab Diffusion Model (Diffusion Limited)

The Slab Diffusion Model, historically associated with H.V. Hempleman and the development of the Royal Navy Physiological Laboratory (RNPL) and British Sub-Aqua Club (BSAC) 88 tables, operates on a fundamentally different physical premise. It challenges the assumption that tissues are well-stirred perfusion baths. Instead, it posits that for certain tissues—particularly those involved in Type I DCS (pain-only bends)—gas exchange is limited by the linear bulk diffusion of gas from the capillary into a tissue mass, modeled as a "slab".

Mathematically, this system is described by Fick's Second Law of Diffusion in a one-dimensional medium. Unlike the exponential nature of perfusion models, diffusion into a semi-infinite slab initially follows a square-root-of-time relationship ($t^{1/2}$). Hempleman's early empirical work suggested that for no-decompression limits, the relationship between depth ($P$) and time ($t$) was constant:

$$P \cdot \sqrt{t} = C$$

This "Root-t" concept implies different safety margins than exponential models. In a slab model, gas does not just "fill" a compartment; it migrates through layers of the slab. During compression (descent), gas diffuses inward from the face of the slab. During decompression (ascent), the gas in the deeper layers of the slab cannot instantly exit; it must diffuse back out through the layers closer to the surface. This creates a "reservoir" effect where deep tissue layers continue to on-gas even as the diver ascends, provided the gradient still favors inward diffusion.

### 2.3 Mathematical Implications for Simulation

The divergence between exponential (Bühlmann) and root (Slab) kinetics dictates the design of the random profile generator.

| Feature           | Bühlmann (Exponential)                               | Slab (Diffusion)                                                         | Simulation Strategy                                                                |
| ----------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Uptake Speed      | Rapid initial uptake, slowing over time.             | Initially rapid ($t^{1/2}$), but penetrating deep layers takes longer.   | Generate short, deep "bounce" dives to test initial uptake divergence.             |
| Off-gassing       | Symmetric to uptake (assuming no bubble formation).  | Asymmetric; gas deep in the slab is "trapped" by outer layers.           | Generate multi-level profiles with "sawtooth" patterns to trap gas in slab models. |
| Repetitive Diving | Compartments clear independently based on $T_{1/2}$. | Residual gas profile in the slab affects subsequent diffusion gradients. | Simulate "dive series" (e.g., 3 dives/day) rather than single dives.               |

Understanding these differences ensures that the "thousands of random profiles" generated are not just random noise, but targeted stress tests that probe the specific physiological disagreements between the two theories.

## 3. Analysis of Open Source Computational Libraries

To implement this backtesting framework, the user identified three libraries. This section analyzes their internal architecture, suitability for high-throughput simulation, and specific utility in handling the data formats required for Slab model validation.

### 3.1 AquaBSD/libbuhlmann: The Simulation Control Engine

**Repository:** https://github.com/AquaBSD/libbuhlmann

libbuhlmann acts as the deterministic oracle for the project. Since the goal is to backtest a new model against an established one, libbuhlmann provides the established baseline. It allows the researcher to ask: "According to the standard currently used in dive computers, is this profile safe?"

#### 3.1.1 Architecture and API Integration

The library is written in C, which offers the performance necessary for running millions of iterations. The source structure typically separates the core logic (src/dive.c) from the interface. The snippet reveals a critical utility: gen_dive.py. This Python script is designed to generate dive profile data and pipe it into the compiled C binary.

The command usage `python test/gen_dive.py -d 20 -t 5 | src/dive` suggests a text-stream input architecture. This is highly advantageous for batch processing. Instead of writing complex C bindings (ctypes/CFFI) to link the library directly to the Slab model (which might be written in Python or MATLAB), the researcher can treat the src/dive binary as a black box function $f(profile) \rightarrow safety$.

#### 3.1.2 Limitations and Workarounds

The library appears to output a decompression schedule (stops) rather than raw tissue tensions. For the Slab model comparison, knowing only the stops is useful but incomplete. Ideal backtesting requires comparing the internal state—the Bühlmann tissue tensions vs. the Slab integrated pressure.

To address this, the researcher may need to slightly modify src/dive.c to print the final pressures of the 16 compartments to STDOUT alongside the deco schedule. Since the library is open source, inserting a printf statement at the end of the calculation loop is a trivial yet high-value modification that exposes the hidden variables needed for deep statistical comparison.

### 3.2 libdivecomputer: The Universal Hardware Interface

**Repository:** https://github.com/libdivecomputer/libdivecomputer

While libbuhlmann deals in theory, libdivecomputer deals in the messy reality of hardware data. It is the industry standard for communicating with dive computers, supporting manufacturers like Suunto, Shearwater, Mares, and Oceanic.

#### 3.2.1 The Data Parsing Architecture

The library's architecture is bifurcated into Transport (I/O) and Parsing.

- **Transport:** Handles USB/Bluetooth protocols to dump memory from the device.
- **Parsing:** Converts that proprietary memory dump into standard units.

For this research, the Parsing API (dc_parser_t) is paramount. When mining data from the web (Section 5), the researcher will encounter raw binary dumps (.bin or .dmp files). These files are unintelligible without the parser. The snippet outlines the C-style pseudo-code for parsing:

```c
parser = <model>_parser_create (parameters);
parser_set_data (parser, data, size);
while (parser_sample_step (parser)) {
    // extract time, depth, temperature
}
```

This confirms that if we possess the binary files, we can extract high-fidelity time-series data (depth every 2-10 seconds). This granularity is essential for the Slab model, which relies on integrating diffusion gradients over time; coarse summary data (e.g., "Max Depth: 30m, Time: 40min") is insufficient for diffusion analysis.

#### 3.2.2 The dctool Utility

The library includes a command-line utility called dctool. This tool allows for the manipulation of dive computer data without writing custom C code.

- **Key Command:** `dctool extract -f <device_name> -o output.xml input.bin`
- **Research Application:** This tool enables "Headless" batch processing. A Python script can iterate through a folder of 10,000 downloaded binary files, invoke dctool extract on each, and generate 10,000 standardized XML files. This creates a massive, standardized dataset from a chaotic collection of proprietary dumps.

### 3.3 Subsurface: The Data Management and Normalization Layer

**Repository:** https://github.com/subsurface/subsurface

Subsurface is a comprehensive logbook application that sits on top of libdivecomputer. It provides the schema and the conversion logic required to normalize the data.

#### 3.3.1 The Subsurface XML Schema

Subsurface uses a documented XML format to store dives. This schema is the logical choice for the "Common Data Format" of this research project.

- **Structure:** It captures not just depth/time profiles, but also gas mixes (Nitrox/Trimix), tank sizes, and environmental variables (temperature).
- **Relevance:** The Slab model's diffusion rate is dependent on the partial pressure of inert gas, which is a function of both depth and the breathing gas mix ($F_{N2}, F_{He}$). The Subsurface XML preserves this gas switch data (`<event type='gaschange'... />`), which is often lost in simpler CSV exports.

#### 3.3.2 Headless Conversion Capabilities

Subsurface supports command-line arguments for import and export.

- **Import:** Can ingest UDDF, CSV, DAN DL7, and proprietary database files (e.g., Shearwater .db, Suunto .sde).
- **Export:** Can output to CSV, XML, or UDDF.
- **Workflow:** The research pipeline can utilize the Subsurface binary as a "File Converter." `subsurface --import=unknown_format.uddf --export=standardized_profile.csv` This capability allows the researcher to accept data from any source—GitHub issue attachment, forum upload, or Kaggle dataset—and funnel it into a single processing pipeline for the Slab model backtester.

## 4. Methodology: Stochastic Simulation and Backtesting

The core requirement is to "generate thousands of random dive profiles" to backtest the new Slab model. This section details the algorithmic design of the generator and the comparative analysis engine.

### 4.1 Designing the Stochastic Diver Algorithm

Generating "random" numbers for depth and time is insufficient; the profiles must be physically possible and physiologically relevant. A "Random Walk" approach with physiological constraints is recommended.

#### 4.1.1 Profile Topology Generation

The simulation should produce three distinct classes of profiles to test different aspects of the Slab theory:

1. **Square Profiles (Calibration):**
   - **Logic:** Descent to depth $D$, hold for time $T$, ascent.
   - **Purpose:** Calibrate the Slab model against known Bühlmann NDLs. The Slab model should roughly converge with Bühlmann for standard recreational dives (e.g., 30m for 20min).
   - **Variables:** $D \sim U(10, 60)$ meters, $T \sim U(5, 60)$ minutes.

2. **Bounce Dives (Diffusion Stress):**
   - **Logic:** Rapid descent to deep depth, very short bottom time, rapid ascent.
   - **Purpose:** Test the $t^{1/2}$ uptake. Diffusion models often permit deeper, shorter exposures than perfusion models because the "slab" takes time to absorb gas.
   - **Variables:** $D \sim U(40, 100)$ meters, $T \sim U(2, 10)$ minutes.

3. **Sawtooth/Multi-level (Hysteresis Stress):**
   - **Logic:** Descent to deep depth, partial ascent, re-descent, final ascent.
   - **Purpose:** Test the "trapping" effect. In the Slab model, gas absorbed deep in the slab during the first spike must diffuse through the slab to exit. A second descent reverses the gradient, potentially trapping the gas. This is where Slab and Bühlmann often diverge most significantly.

#### 4.1.2 Implementation with Python and AquaBSD/libbuhlmann

The generator script (Python) acts as the coordinator.

```python
import subprocess
import random
import pandas as pd

def generate_sawtooth_profile():
    # Logic to create a list of (time, depth) tuples
    # alternating between deep and shallow depths
    profile = []
    current_depth = 0
    #... random walk logic...
    return profile

def calculate_buhlmann_risk(profile_data):
    # Pipe to libbuhlmann
    process = subprocess.Popen(
        ['./src/dive'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )
    stdout, _ = process.communicate(input=profile_data)
    # Parse output for deco stops or NDL
    return parse_buhlmann_output(stdout)

def calculate_slab_risk(profile_data):
    # Python implementation of Finite Difference Method for Slab
    # Solve dP/dt = D * d^2P/dx^2
    #...
    return max_slab_tension
```

### 4.2 The Backtesting Comparator

For each generated profile ($N=50,000+$), the system calculates two risk metrics:

1. **Bühlmann Supersaturation:** The maximum percentage of the M-value reached in the leading compartment.
2. **Slab Supersaturation:** The maximum gradient reached in the tissue slab relative to the critical limit.

**The Divergence Matrix:**

The results should be plotted on a matrix where:

- **X-axis:** Dive Depth
- **Y-axis:** Bottom Time
- **Color:** $\Delta Risk = (Risk_{Slab} - Risk_{Buhlmann})$

**Interpretation:**

- **Red Zones** ($\Delta > 0$): The Slab model is more conservative (predicts DCS where Bühlmann says safe). These are potential areas where current algorithms might be aggressive.
- **Blue Zones** ($\Delta < 0$): The Slab model is more aggressive. These are areas where the Slab model might allow profiles that modern safety standards prohibit.

## 5. Programmatic Data Mining Strategy

The request asks for a method to "get real dive log data either from github issues or forums programatically." This relies on the behavior of open-source communities where users upload failed log files for debugging.

### 5.1 Target Data Sources

1. **GitHub Issues (Subsurface & libdivecomputer):** Users frequently attach .zip, .xml, or .bin files to issues describing import failures. These files are high-value because they often represent edge cases that broke standard parsers—exactly the kind of "stress test" data needed.

2. **Community Repositories:** The snippet mentions Diveboard and snippet mentions Kaggle datasets. These are structured sources but may lack the raw binary fidelity of GitHub issue attachments.

3. **Forums:** Sites like scubaboard.com or divelogs.de sometimes host shared log files, though programmatic access is harder due to authentication barriers compared to the open GitHub API.

### 5.2 The GitHub Mining Script

To automate this, we leverage the GitHub REST API. The script must search for issues containing attachments and download them.

#### 5.2.1 Search Strategy

GitHub does not have a direct "search by attachment" API. We must iterate through issues and parse the body text for markdown links to file uploads.

- **Target Repos:** `subsurface/subsurface`, `libdivecomputer/libdivecomputer`.
- **Keywords:** "attachment", "dump", "log", ".bin", ".xml".

#### 5.2.2 Python Implementation Details

The script uses the `requests` library. It requires a Personal Access Token (PAT) to handle rate limiting (5000 requests/hour).

**Key Logic Block:**

1. **Fetch Issues:** `GET /repos/{owner}/{repo}/issues?state=all`
2. **Scan Body:** Use Regex `r'https://github\.com/.*?/files/.*?'` to find attachment URLs.
3. **Filter Extensions:** Only download `.xml` (Subsurface), `.uddf`, `.bin` (raw dumps), `.sde`, `.db`.
4. **Metadata Tagging:** Save the file with the Issue ID (e.g., `issue_1245_dump.bin`). This allows traceability; if a file produces a weird result in the Slab model, the researcher can go back to the GitHub issue to read the context (e.g., "Diver ascended rapidly due to equipment failure").

#### 5.2.3 Handling "Orphaned" Binary Files

Files downloaded from libdivecomputer issues are often raw binary dumps without metadata telling you which computer model created them.

- **Solution:** The dctool utility supports a "probe" or "extract" mode. The mining script can attempt to parse an unknown binary file by iterating through common device descriptors. If dctool successfully generates an XML output with valid timestamps, the file is accepted into the dataset.

### 5.3 Alternative Data Sources (Google Research)

The prompt asks to research "other sources."

- **Kaggle:** Snippet identifies a "Scuba Diving Logbook" dataset on Kaggle. Analysis shows this contains columns like "Depth", "Bottom Time", but likely lacks the second-by-second profile data required for diffusion modeling. It serves as a source for demographic distribution (e.g., typical depth ranges) to tune the stochastic generator but cannot replace raw logs.
- **Diveboard:** Snippet mentions diveboard.com exporting GBIF datasets. These are primarily biological observations (fish sightings) but contain geolocation and depth data, useful for spatial analysis of diving intensity.
- **R Packages:** Snippet mentions divewatchr, an R package for visualizing dive logs from Google Sheets. This indicates that scraping public Google Sheets (if indexed) could be another vector, though lower yield than GitHub.

## 6. Synthesis and Implementation Roadmap

The proposed research project combines theoretical simulation with data mining to validate the Slab Diffusion Model.

### 6.1 Phase 1: Infrastructure Setup

1. **Compile libbuhlmann:** Ensure src/dive is executable and gen_dive.py is functional.
2. **Compile libdivecomputer:** Build the dctool binary for command-line parsing.
3. **Install Subsurface:** Ensure the CLI subsurface command is available for format conversion.

### 6.2 Phase 2: Data Mining

1. Run the GitHub Mining Script targeting `subsurface/subsurface` and `libdivecomputer/libdivecomputer`.
2. Target yield: ~5,000 files.
3. Run dctool and subsurface conversion scripts to normalize these into a standardized CSV format (Time, Depth).

### 6.3 Phase 3: Stochastic Simulation

1. Develop the Python Generator to create 50,000 profiles (Square, Bounce, Sawtooth).
2. Run these profiles through libbuhlmann to label them (Safe/Unsafe).

### 6.4 Phase 4: Slab Model Backtesting

1. Implement the Finite Difference Solver for the Slab equation in Python.
2. Process both the Mined Data (Real) and Simulated Data (Synthetic) through the Slab model.
3. Identify profiles where Slab Tension > Critical Limit but Bühlmann = Safe (Type I Divergence).
4. Identify profiles where Slab Tension < Critical Limit but Bühlmann = Unsafe (Type II Divergence).

### 6.5 Conclusion

This framework utilizes the distinct strengths of each library: libbuhlmann as the theoretical control, libdivecomputer as the technical key to lock proprietary data, and Subsurface as the data warehouse. By rigorously mining GitHub issues, the research transforms software debugging artifacts into a valuable repository of human physiological data, providing a unique empirical anchor for the theoretical Slab Diffusion Model.

## 7. Tables and Structured Data

### Table 1: Comparative Analysis of Target Libraries

| Library             | Primary Language    | Key Research Utility                                                   | Input Data Format            | Output Data Format                 |
| ------------------- | ------------------- | ---------------------------------------------------------------------- | ---------------------------- | ---------------------------------- |
| AquaBSD/libbuhlmann | C (Python bindings) | Oracle / Control: Generates verified ZH-L16C safety schedules.         | Text stream (Depth/Time)     | Decompression Stops / NDL          |
| libdivecomputer     | C                   | Parser: Unlocks proprietary binary dump files from hardware.           | Binary Blobs (.bin, .dmp)    | Parsed Samples (Time, Depth, Temp) |
| Subsurface          | C++ / Qt            | Normalizer: Converts diverse formats (UDDF, FIT, DL7) to standard XML. | Various (UDDF, XML, DB, FIT) | Standardized XML or CSV            |

### Table 2: Model Physics Comparison

| Feature             | Bühlmann ZH-L16C                               | Slab Diffusion Model (Hempleman)                          |
| ------------------- | ---------------------------------------------- | --------------------------------------------------------- |
| Limiting Mechanism  | Perfusion: Blood flow limits gas exchange.     | Diffusion: Transport through tissue bulk limits exchange. |
| Uptake Function     | Exponential: $P = P_0 + \Delta P(1 - e^{-kt})$ | Root-Time: $P \propto t^{1/2}$ (initially).               |
| Compartments        | 16 Parallel Tissues (Independent).             | Single Finite Slab (Interdependent layers).               |
| Off-Gassing         | Symmetric to uptake (assuming no bubbles).     | Asymmetric; deep gas must diffuse through outer layers.   |
| Critical Divergence | Long, shallow exposures (saturation).          | Short, deep "bounce" dives & rapid repetitive dives.      |

### Table 3: Data Mining Source Evaluation

| Source                          | Data Type       | Accessibility        | Relevance to Slab Validation                                                         |
| ------------------------------- | --------------- | -------------------- | ------------------------------------------------------------------------------------ |
| GitHub Issues (libdivecomputer) | Binary Dumps    | High (Public API)    | Critical: Contains raw, high-frequency samples needed for diffusion math.            |
| GitHub Issues (subsurface)      | XML / UDDF / DB | High (Public API)    | High: Often contains problem files that stress-test parsers and models.              |
| Kaggle (Scuba Logbook)          | CSV / Tabular   | High (Download)      | Low: Often lacks second-by-second profile data; good for demographics only.          |
| Diveboard                       | GBIF / Export   | Medium (Request/API) | Medium: Good for volume, but data quality varies (manual entry vs. computer upload). |

## 8. Detailed Analysis of Data Mining Implementation

The extraction of data from GitHub requires specific attention to the handling of attachments. Since the GitHub API does not provide a direct "download attachments" endpoint for issues, the following regex-based strategy is required within the Python scripting environment.

### 8.1 Script Logic for Attachment Extraction

The script must iterate through the JSON response of the Issues API. The body field of the issue contains the user's description, often including the file link.

**Regex Pattern:**

```
\[.*?\]\((https://github\.com/[^/]+/[^/]+/files/\d+/[^)]+)\)
```

This pattern captures the standard Markdown link format used by GitHub for file uploads. The script should extract the URL group, check the file extension against a whitelist (.xml, .bin, .zip, .uddf), and execute a GET request to retrieve the file.

### 8.2 Handling Rate Limits

GitHub's API enforces a strict limit of 5000 requests per hour for authenticated users.

- **Strategy:** The script must inspect the `X-RateLimit-Remaining` header in every response.
- **Backoff:** If the remaining count drops below 100, the script should pause execution (`time.sleep()`) until the `X-RateLimit-Reset` timestamp is reached. This ensures the mining process can run continuously over 24-48 hours to harvest the entire history of the repositories without being banned.

### 8.3 Privacy and Ethics

While the data is public, it may contain personal identifiers (names, GPS coordinates of home addresses).

- **Sanitization:** Before feeding the mined data into the backtesting model, a sanitization pass using Subsurface or a custom XML parser is necessary to remove `<location>`, `<buddy>`, and `<notes>` tags, retaining only the time/depth/pressure profile data required for physiological modeling. This creates an anonymized research dataset safe for long-term storage and analysis.
