import io
import requests
import zipfile
import MeCab
import pandas as pd
import streamlit as st
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from masking import masking
import numpy as np
from PIL import Image
import cv2

# フォントファイルのURL
font_url = "https://moji.or.jp/wp-content/ipafont/IPAexfont/ipaexg00401.zip"

# フォントファイルをダウンロード
response = requests.get(font_url)

# フォントファイルを解凍
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall()

# ページのレイアウトを設定
st.set_page_config(
    page_title="テキスト可視化",
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトルの設定
st.title("テキスト可視化")

# サイドバーにアップロードファイルのウィジェットを表示
st.sidebar.markdown("# ファイルアップロード")
uploaded_file = st.sidebar.file_uploader(
    "テキストファイルをアップロードしてください", type="txt"
)

# ワードクラウド、出現頻度表の各処理をサイドバーに表示
st.sidebar.markdown("# 可視化のオプション")
if uploaded_file is not None:
    # 処理の選択
    option = st.sidebar.selectbox(
        "処理の種類を選択してください", ["ワードクラウド", "マスクワードクラウド", "出現頻度表"]
    )
    # ワードクラウドの表示
    if option == "ワードクラウド":
        pos_options = ["名詞", "形容詞", "動詞", "副詞", "助詞", "助動詞", "接続詞", "感動詞", "連体詞", "記号", "未知語"]
        # マルチセレクトボックス
        selected_pos = st.sidebar.multiselect("品詞選択", pos_options, default=["名詞"])
        if st.sidebar.button("生成"):
            st.markdown("## ワードクラウド")
            with st.spinner("Generating..."):
                io_string = io.StringIO(uploaded_file.getvalue().decode("shift-jis"))
                text = io_string.read()
                tagger = MeCab.Tagger('-r/dev/null -d/home/hoge/mydic')
                node = tagger.parseToNode(text)
                words = []
                while node:
                    if node.surface.strip() != "":
                        word_type = node.feature.split(",")[0]
                        if word_type in selected_pos:  # 対象外の品詞はスキップ
                            words.append(node.surface)
                    node = node.next
                word_count = Counter(words)
                wc = WordCloud(
                    width=800,
                    height=800,
                    background_color="white",
                    font_path="./ipaexg00401/ipaexg.ttf",  # Fontを指定
                )
                # ワードクラウドを作成
                wc.generate_from_frequencies(word_count)
                # ワードクラウドを表示
                st.image(wc.to_array())
    
    # 出現頻度表の表示
    elif option == "出現頻度表":
        pos_options = ["名詞", "形容詞", "動詞", "副詞", "助詞", "助動詞", "接続詞", "感動詞", "連体詞", "記号", "未知語"]
        # マルチセレクトボックス
        selected_pos = st.sidebar.multiselect("品詞選択", pos_options, default=pos_options)
        if st.sidebar.button("生成"):
            st.markdown("## 出現頻度表")
            with st.spinner("Generating..."):
                io_string = io.StringIO(uploaded_file.getvalue().decode("shift-jis"))
                text = io_string.read()
                tagger = MeCab.Tagger('-r/dev/null -d/home/hoge/mydic')
                node = tagger.parseToNode(text)

                # 品詞ごとに出現単語と出現回数をカウント
                pos_word_count_dict = {}
                while node:
                    pos = node.feature.split(",")[0]
                    if pos in selected_pos:
                        if pos not in pos_word_count_dict:
                            pos_word_count_dict[pos] = {}
                        if node.surface.strip() != "":
                            word = node.surface
                            if word not in pos_word_count_dict[pos]:
                                pos_word_count_dict[pos][word] = 1
                            else:
                                pos_word_count_dict[pos][word] += 1
                    node = node.next

                # カウント結果を表にまとめる
                pos_dfs = []
                for pos in selected_pos:
                    if pos in pos_word_count_dict:
                        df = pd.DataFrame.from_dict(pos_word_count_dict[pos], orient="index", columns=["出現回数"])
                        df.index.name = "出現単語"
                        df = df.sort_values("出現回数", ascending=False)
                        pos_dfs.append((pos, df))

                # 表を表示
                for pos, df in pos_dfs:
                    st.write(f"【{pos}】")
                    st.dataframe(df, 400, 400)
                   
    elif option == "マスクワードクラウド":
        uploaded_original_img_file = st.sidebar.file_uploader(
            "画像ファイルをアップロードしてください", type="jpg"
        )
        pos_options = ["名詞", "形容詞", "動詞", "副詞", "助詞", "助動詞", "接続詞", "感動詞", "連体詞", "記号", "未知語"]
        # マルチセレクトボックス
        selected_pos = st.sidebar.multiselect("品詞選択", pos_options, default=["名詞"])
       
        if uploaded_original_img_file is not None:
            if st.sidebar.button("生成"):
                st.markdown("## マスクワードクラウド")
                
                with st.spinner("Generating..."):
                    io_string = io.StringIO(uploaded_file.getvalue().decode("shift-jis"))
                    text = io_string.read()
                    tagger = MeCab.Tagger('-r/dev/null -d/home/hoge/mydic')
                    node = tagger.parseToNode(text)
                    words = []
                    while node:
                        if node.surface.strip() != "":
                            word_type = node.feature.split(",")[0]
                            if word_type in selected_pos: # 対象外の品詞はスキップ
                                words.append(node.surface)
                        node = node.next
                    word_count = Counter(words)
                    
                    # アップロードされた画像を読み込む
                    uploaded_img = Image.open(uploaded_original_img_file)
                    
                    # numpy配列に画像を変換
                    fg_img = np.array(uploaded_img)
                    
                    # RGBに変換
                    fg_img = cv2.cvtColor(fg_img, cv2.COLOR_RGB2BGR)
                    
                    # マスクした画像を出力
                    masked_img = masking(fg_img)
                    masked_img_array = np.array(masked_img)
                    stopwords = set(STOPWORDS)
                    stopwords.add("said")
                    
                    # マスクした画像でwcオブジェクトを作成
                    wc = WordCloud(
                        background_color="white",
                        max_words=2000,
                        width=800,
                        height=800,
                        mask=masked_img_array,
                        stopwords=stopwords,
                        contour_width=3,
                        contour_color='steelblue',
                        font_path="./ipaexg00401/ipaexg.ttf",  # Fontを指定
                        )
                    
                    wc.generate_from_frequencies(word_count)
                    
                    # wc画像を表示
                    st.image(wc.to_array())
               
else:
    # テキスト未アップロード時の処理
    st.write("テキストファイルをアップロードしてください。")
