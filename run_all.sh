#!/bin/bash

# 🎬 Movie Recommender System - Automated Setup & Execution Script
# This script automates: conda env creation → pipeline execution → streamlit launch

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="mf_env"
ENV_FILE="environment.yml"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🎬 Movie Recommender System - Auto Setup${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Step 1: Check if conda is installed
echo -e "${YELLOW}[1/6] Checking conda installation...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Error: conda is not installed or not in PATH${NC}"
    echo -e "${RED}Please install Anaconda or Miniconda first${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Conda found: $(conda --version)${NC}\n"

# Step 2: Create conda environment (if it doesn't exist)
echo -e "${YELLOW}[2/6] Setting up conda environment...${NC}"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${GREEN}✅ Environment '${ENV_NAME}' already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n ${ENV_NAME} -y
        echo -e "${YELLOW}Creating new environment from ${ENV_FILE}...${NC}"
        conda env create -f ${ENV_FILE}
        echo -e "${GREEN}✅ Environment recreated${NC}"
    fi
else
    echo -e "${YELLOW}Creating environment from ${ENV_FILE}...${NC}"
    conda env create -f ${ENV_FILE}
    echo -e "${GREEN}✅ Environment created successfully${NC}"
fi
echo ""

# Step 3: Preprocess data
echo -e "${YELLOW}[3/6] Preprocessing data (train/test split, matrices)...${NC}"
conda run -n ${ENV_NAME} python -m utils.preprocess
echo -e "${GREEN}✅ Data preprocessing complete${NC}\n"

# Step 4: Train SVD model
echo -e "${YELLOW}[4/6] Training SVD baseline model...${NC}"
conda run -n ${ENV_NAME} python scripts/train_svd.py
echo -e "${GREEN}✅ SVD model trained${NC}\n"

# Step 5: Train PMF model with demographics
echo -e "${YELLOW}[5/6] Training PMF model with demographic features...${NC}"
conda run -n ${ENV_NAME} python scripts/train_pmf_bias.py
echo -e "${GREEN}✅ PMF model trained${NC}\n"

# Step 6: Generate visualizations (optional but recommended)
echo -e "${YELLOW}[6/6] Generating evaluation visualizations...${NC}"
conda run -n ${ENV_NAME} python scripts/generate_visualizations.py
echo -e "${GREEN}✅ Visualizations generated${NC}\n"

# Success summary
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✅ All pipeline steps completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}\n"

# Launch Streamlit
echo -e "${BLUE}🚀 Launching Streamlit dashboard...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}\n"

# Launch streamlit (this will keep running)
conda run -n ${ENV_NAME} streamlit run app.py
