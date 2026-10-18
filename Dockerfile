# ==========================================
# Stage 1: Build the C# projects
# ==========================================
ARG REGISTRY=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/mcr.microsoft.com
FROM ${REGISTRY}/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy dependency files and restore
COPY ["SolTTS.Web/SolTTS.Web.csproj", "SolTTS.Web/"]
COPY ["SolTTS.Core/SolTTS.Core.csproj", "SolTTS.Core/"]
COPY ["SolTTS.CLI/SolTTS.CLI.csproj", "SolTTS.CLI/"]
RUN dotnet restore "SolTTS.Web/SolTTS.Web.csproj"

# Copy all source code
COPY . .

# Compile and publish the Web project
WORKDIR "/src/SolTTS.Web"
RUN dotnet publish "SolTTS.Web.csproj" -c Release -o /app/publish /p:UseAppHost=false

# Prepare configuration files and cross-platform tool scripts
WORKDIR "/src"
RUN cp config.template.yaml /app/publish/config.template.yaml && \
    cp config.template.yaml /app/publish/config.yaml && \
    # Copy the tools/converter directory (removing the unused Windows runtime)
    mkdir -p /app/publish/tools/converter && \
    cp -r tools/converter/v1_converter.py /app/publish/tools/converter/ && \
    cp -r tools/converter/templates /app/publish/tools/converter/

# ==========================================
# Stage 2: Final runtime environment (.NET 10, Python 3.12, and model resources)
# ==========================================
ARG REGISTRY=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/mcr.microsoft.com
FROM ${REGISTRY}/dotnet/aspnet:10.0 AS final
WORKDIR /app

# Install dependencies (for Ubuntu Noble)
# 1. System dependencies
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources || true && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python environment and dependencies (force CPU-only torch to reduce image size by several GB)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    onnx numpy onnxsim onnxruntime && \
    pip install --no-cache-dir torch -f https://mirrors.aliyun.com/pytorch-wheels/cpu

# Copy the compiled application from the build stage
COPY --from=build /app/publish .

# Copy the core resource folder (avoid duplicating it during the build stage)
COPY resources-minimal /app/resources

# Expose the port
EXPOSE 5000
ENV ASPNETCORE_URLS=http://+:5000

# Run the Web service
ENTRYPOINT ["dotnet", "sol-server.dll"]
