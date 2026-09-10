# Techno-Economic Application Frameworks for BESS

This repository contains the source code for the paper *"Techno-Economic Application Frameworks for Battery Energy Storage Systems (BESS) in East Africa's Industrial and Renewable Energy Sectors"*. 

It features `bess_feasibility.py`, a self-contained, object-oriented Python algorithm designed to evaluate the financial and operational viability of BESS. The model calculates 10-year discounted cash flows, Levelized Cost of Storage (LCOS), and unified sizing without relying on computationally burdensome 8,760-hour time-series datasets.

## Key Modeling Modules Included:

*   **Peak Shaving:** Calculates capacity relief and payback from clipping inductive load spikes.
*   **Time-of-Use (TOU) Arbitrage:** Evaluates the net margin per kWh shifted across regional tariff bands.
*   **Diesel Displacement:** Models the substitution of high-cost fossil fuel backup generation with Solar+BESS hybrid systems.
*   **Industrial UPS:** Quantifies the financial value of zero-latency continuity and spoilage prevention during grid outages. 
