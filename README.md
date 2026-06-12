# UEBA - Insider Threat Detection System

UEBA (User and Entity Behavior Analytics) is a system designed to detect insider threats using machine learning and behavioral analysis on CERT r4.2 datasets (Logon, HTTP, Device, Email, and File activity).

## 🚀 Key Features

-   **Multi-Source Data Ingestion**: Processes security logs from various organizational sources.
-   **Granular Feature Engineering**: Analyzes activity at the user-hour level across multiple temporal scales (1D, 7D, 30D).
-   **Hybrid Ensemble Model**: Combines supervised ML (Random Forest, Gradient Boosting, MLP) with unsupervised anomaly detection (KMeans, Elliptic Envelope).
-   **Real-time Simulation**: Includes a simulator for generating Windows event logs for testing.
-   **ELK Integration**: Built-in support for Elasticsearch, Logstash, and Kibana for data storage and visualization.

---

## 📂 Project Structure

-   `scripts/`: Core Python logic for data cleaning, feature engineering, and modeling.
-   `configs/`: Docker configuration and application settings.
-   `data/`: Directory for input datasets, models, and outputs.
-   `logs/`: Application and processing logs.

---

## 🐳 Running with Docker

The easiest way to get started is using Docker Compose. It will spin up the entire ELK stack along with the UEBA application.

### Quick Start

```bash
docker-compose -f configs/docker-compose.yml up --build
```

For detailed instructions on profiles (including the Simulator) and troubleshooting, please refer to the **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)**.

---

## 📦 Local Setup (Optional)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
