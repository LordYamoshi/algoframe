# AlgoFrame

**AlgoFrame** is a desktop trading assistant for **Warframe** built around smarter market analysis, inventory management, and automated trading workflows.

It connects Warframe inventory data with live and historical data from [warframe.market](https://warframe.market) to help identify useful trades, manage listings, track performance, and reduce repetitive market work.

The project is built with **Tauri, Rust, React, and SQLite**.

---

## Overview

Warframe trading usually means repeatedly checking prices, comparing listings, updating orders, and trying to determine whether an item is actually worth buying.

AlgoFrame brings those tasks together into one desktop application.

It is designed around three questions:

> What is worth buying?

> What is worth selling?

> Where should my platinum be allocated?

Rather than relying only on the current lowest or highest listing, AlgoFrame can use market statistics, live orders, inventory information, profit thresholds, price history, and configurable trading rules when evaluating items.

---

## Features

### Live Market Trading

AlgoFrame can monitor and manage supported listings on warframe.market.

It can:

- Create and update WTB orders
- Create and update WTS orders
- Monitor competing listings
- Apply configurable profit requirements
- Respect buy and sell price limits
- Limit inventory accumulation
- Manage available platinum across buy orders
- Detect when listings are no longer worth maintaining

---

### Market Analysis

Trading decisions can use information such as:

- Current buy orders
- Current sell orders
- Historical prices
- Moving averages
- Trading volume
- Profit
- Profit margin
- Supply and demand
- Price movement
- Trading tax
- Existing inventory

AlgoFrame is being developed toward a more advanced opportunity-scoring system that considers not only raw profit, but also liquidity, turnover, competition, price stability, and capital efficiency.

---

### Warframe Inventory

AlgoFrame can analyze tradable inventory and connect owned items directly to market information.

Supported categories include:

- Prime parts
- Prime sets
- Mods
- Arcanes
- Relics
- Rivens
- Syndicate items

Inventory views can display owned quantities, estimated market value, completion state for sets, and available listing actions.

---

### Price Discovery

Not every Warframe item has equally reliable historical pricing.

AlgoFrame can combine multiple pricing sources and retain discovered values locally so inventory analysis does not depend entirely on one dataset.

This makes it possible to evaluate significantly more of an account's inventory without continuously requesting the same market information.

---

### Trade Tracking

Trading data can be stored locally and used to review performance over time.

Examples include:

- Purchases
- Sales
- Revenue
- Expenses
- Profit
- Profit margins
- Item performance
- Price history

This data can also be used later to improve AlgoFrame's trading decisions.

---

## Trading System

AlgoFrame's trading system begins by filtering the wider Warframe market into a smaller set of possible opportunities.

A simplified flow looks like this:

```text
Market Data
    │
    ▼
Candidate Filtering
    │
    ▼
Live Order Analysis
    │
    ▼
Profit / Risk Evaluation
    │
    ▼
Capital Allocation
    │
    ▼
WTB / WTS Management
```

Current filters and rules can include metrics such as volume, profit, margin, price movement, market price, trading tax, and inventory limits.

The long-term direction is to move from simple threshold-based selection toward opportunity scoring.

For example:

```text
Opportunity Score

Expected Profit
× Liquidity
× Turnover
× Execution Confidence
× Capital Efficiency
- Market Risk
```

This allows two items with similar theoretical profit to be treated differently when one trades significantly faster or has a more stable market.

---

## Technology

AlgoFrame uses a lightweight desktop architecture.

**Tauri 2**  
Desktop runtime and native integration.

**Rust**  
Market processing, application services, trading logic, database access, and local integrations.

**React**  
User interface.

**Mantine**  
UI component framework.

**SQLite / SeaORM**  
Local application and trading data.

**TanStack Query**  
Frontend server-state management.

---

## Local Data

AlgoFrame stores application data locally.

The exact directory depends on the application identifier used by the build.

Typical stored data can include:

```text
Database
Settings
Authentication data
Market caches
Inventory caches
Price history
Logs
```

Sensitive account information and locally captured Warframe data should never be committed to the repository.

---

## Development

### Requirements

You will need:

- Node.js
- pnpm
- Rust
- Tauri 2 prerequisites

Install the frontend dependencies:

```bash
pnpm install
```

Run AlgoFrame in development mode:

```bash
pnpm tauri dev
```

Run the frontend separately:

```bash
pnpm dev
```

Create a desktop build:

```bash
pnpm tauri build
```

---

## Project Direction

AlgoFrame is intended to go beyond simply automating listing updates.

Planned areas of improvement include:

- Better WTB opportunity detection
- Smarter WTS pricing
- Liquidity-aware trading
- Order-book analysis
- Price-outlier detection
- Adaptive repricing
- Better platinum allocation
- Inventory opportunity ranking
- Trade-performance feedback
- Improved analytics
- More reliable service handling
- Better diagnostics and debugging tools

The eventual goal is for AlgoFrame to rank trading opportunities by how useful they actually are, rather than only by their theoretical price difference.

---

## Attribution

AlgoFrame is based on the open-source **QuantFrame** project created by **Kenya-DK**:

https://github.com/Kenya-DK/quantframe-react

AlgoFrame is an independently modified project and is not an official QuantFrame release.

---

## Disclaimer

AlgoFrame is an unofficial community project.

It is not affiliated with or endorsed by:

- Digital Extremes
- Warframe
- warframe.market

Warframe and related trademarks are property of Digital Extremes.

Market automation and trading features should be used responsibly.

---

## License

AlgoFrame is derived from software distributed under the **GNU General Public License v3.0**.

See the repository's `LICENSE` file for the full license terms.

Original copyright and attribution requirements remain applicable to modified versions.
