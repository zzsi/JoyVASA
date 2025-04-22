# FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive  apt-get -y install \
    autoconf \
    automake \
    build-essential \
    cmake \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    pkg-config \
    texinfo \
    wget \
    zlib1g-dev \
    nasm \
    yasm \
    libx265-dev \
    libnuma-dev \
    libvpx-dev \
    libmp3lame-dev \
    libopus-dev \
    libx264-dev \
    libfdk-aac-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ~/ffmpeg_sources ~/bin && cd ~/ffmpeg_sources && \
    wget -O ffmpeg-4.2.2.tar.bz2 https://ffmpeg.org/releases/ffmpeg-4.2.2.tar.bz2 && \
    tar xjvf ffmpeg-4.2.2.tar.bz2 && \
    cd ffmpeg-4.2.2 && \
    PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
    --prefix="$HOME/ffmpeg_build" \
    --pkg-config-flags="--static" \
    --extra-cflags="-I$HOME/ffmpeg_build/include" \
    --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
    --extra-libs="-lpthread -lm" \
    --bindir="$HOME/bin" \
    --enable-libfdk-aac \
    --enable-gpl \
    --enable-libass \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-nonfree && \
    PATH="$HOME/bin:$PATH" make -j8 && \
    make install -j8 && \
    hash -r
RUN mv ~/bin/ffmpeg /usr/local/bin && mv ~/bin/ffprobe /usr/local/bin && mv ~/bin/ffplay /usr/local/bin
# Create an alias for ffmpeg
RUN echo "alias ffmpeg='/usr/local/bin/ffmpeg'" >> ~/.bashrc  # this did not seem to work

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install fairseq
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install git+https://github.com/kungfuai/CVlization.git
RUN pip install wandb
# this is to copy the model weights and code
# COPY . .
