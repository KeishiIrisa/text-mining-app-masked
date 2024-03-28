FROM python:3.9

# mecabのインストール
RUN apt-get update && apt-get install -y \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    git \
    make \
    curl \
    xz-utils \
    file \
    sudo \
    wget \
    libgl1-mesa-dev

# neologdのインストール
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && \
    cd mecab-ipadic-neologd && \
    ./bin/install-mecab-ipadic-neologd -n -y && \
    echo dicdir = `mecab-config --dicdir`"/mecab-ipadic-neologd">/etc/mecabrc && \
    cp /etc/mecabrc /usr/local/etc

WORKDIR /opt/app-root/src
COPY ./src /opt/app-root/src

# フォントのダウンロード
RUN wget https://moji.or.jp/wp-content/ipafont/IPAexfont/ipaexg00401.zip
RUN unzip ipaexg00401.zip && rm -f ipaexg00401.zip

# ライブラリのインストール
RUN pip install streamlit mecab-python3 wordcloud opencv-python-headless

