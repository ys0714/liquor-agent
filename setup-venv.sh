#!/bin/bash
# ============================================================================
# 黑马 AI LLM RAG Agent 项目 - Python venv 环境配置脚本
# HeiMa AI LLM RAG Agent Project - Python venv Environment Setup Script
# ============================================================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "  黑马 AI LLM RAG Agent 环境配置脚本"
echo "  HeiMa AI LLM RAG Agent Env Setup"
echo "========================================"

# 设置环境名称和路径
ENV_NAME="hei-mai-llm-rag"
ENV_DIR="$(pwd)/.venv-${ENV_NAME}"

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ 检测到 Python ${PYTHON_VERSION}"

# 创建虚拟环境
if [ -d "${ENV_DIR}" ]; then
    echo "警告：虚拟环境已存在于 ${ENV_DIR}"
    echo "Warning: Virtual environment already exists at ${ENV_DIR}"
    read -p "是否删除并重新创建？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在删除旧环境..."
        echo "Removing old environment..."
        rm -rf "${ENV_DIR}"
    else
        echo "========================================"
        echo "  环境已存在，跳过创建"
        echo "  Environment exists, skipping creation"
        echo "========================================"
        echo ""
        echo "激活环境 / Activate environment:"
        echo "  source ${ENV_DIR}/bin/activate"
        echo ""
        exit 0
    fi
fi

echo "正在创建 Python 虚拟环境 '${ENV_NAME}'..."
echo "Creating Python virtual environment '${ENV_NAME}'..."

python3 -m venv "${ENV_DIR}"

# 激活虚拟环境
echo "激活虚拟环境..."
echo "Activating virtual environment..."
source "${ENV_DIR}/bin/activate"

# 升级 pip
echo "升级 pip..."
echo "Upgrading pip..."
pip install --upgrade pip

# 安装依赖
echo "安装项目依赖..."
echo "Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "  环境配置完成！"
echo "  Environment setup completed!"
echo "========================================"
echo ""
echo "激活环境 / Activate environment:"
echo "  source ${ENV_DIR}/bin/activate"
echo ""
echo "配置 .env 文件 / Configure .env file:"
echo "  cp .env.example .env"
echo "  然后编辑 .env 文件，填入你的阿里云 DashScope API Key"
echo "  Then edit .env and fill in your Alibaba Cloud DashScope API Key"
echo ""
