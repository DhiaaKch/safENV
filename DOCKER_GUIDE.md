# Docker Guide - UEBA Insider Threat Detection

This guide explains how to build and run the UEBA project using Docker.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- [Docker Compose](https://docs.docker.com/compose/install/) (included in Docker Desktop for Windows/Mac).

## Project Overview (Docker)

The project uses a multi-service architecture:

1.  **Elasticsearch**: Database for storing and searching security events.
2.  **Kibana**: Visual interface for analyzing data and viewing dashboards.
3.  **Logstash**: Data processing pipeline that ingests logs and sends them to Elasticsearch.
4.  **UEBA App**: The core Python application for feature engineering and ML prediction.
5.  **UEBA Simulator**: Optional service to generate real-time log activity for demonstration.

---

## 🚀 Getting Started

### 1. Build and Start the Environment

Run the following command from the **project root** to start Elasticsearch, Logstash, Kibana, and the core App:

```bash
docker-compose -f configs/docker-compose.yml up --build
```

### 2. Run with Simulator (Optional)

If you want to simulate live Windows logs being sent to the stack, use the `simulator` profile:

```bash
docker-compose -f configs/docker-compose.yml --profile simulator up --build
```

### 3. Service Access

Once started, you can access the following services:

| Service | URL | Default Port |
| :--- | :--- | :--- |
| **Kibana** | [http://localhost:5601](http://localhost:5601) | 5601 |
| **Elasticsearch** | [http://localhost:9200](http://localhost:9200) | 9200 |
| **Logstash** | TCP/UDP | 5000 / 5514 |

---

## 🛠️ Common Commands

### Stop and Remove Containers
```bash
docker-compose -f configs/docker-compose.yml down
```

### View Logs
```bash
docker-compose -f configs/docker-compose.yml logs -f [service_name]
```
*(Example: `docker-compose -f configs/docker-compose.yml logs -f ueba-app`)*

### Run a Shell in the App Container
```bash
docker-compose -f configs/docker-compose.yml exec ueba-app bash
```

---

## 📄 File Persistence

-   **Elasticsearch Data**: Persisted in a Docker volume named `elasticsearch-data`.
-   **App Code**: Mounted as a volume from the project root (`..:/app`), allowing you to edit code locally and see changes in the container without rebuilding (except for dependency changes).

## ⚠️ Troubleshooting

-   **Memory**: Elasticsearch requires at least 4GB of RAM assigned to Docker on Windows. If it crashes, check your Docker Desktop settings.
-   **Logstash Connection**: If the simulator can't connect to Logstash, ensure Logstash is fully started before running the simulator.
