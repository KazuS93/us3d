import streamlit as st
import numpy as np
import cv2
from skimage import filters, measure, morphology
import plotly.graph_objects as go
from io import BytesIO
import tempfile
import os

# ------------------------
# 前処理 & 骨抽出ロジック
# ------------------------

def preprocess_frame(gray):
    # ガウシアンノイズ除去
    denoised = filters.gaussian(gray, sigma=1.0)
    denoised = (denoised * 255).astype(np.uint8)
    # コントラスト強調（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

def frames_to_volume(frames, step_mm=0.5):
    """
    2Dフレーム列 → 3Dボリューム
    frames: [H, W]のnumpy配列リスト
    step_mm: フレーム間距離（mm）
    """
    vol = np.stack(frames, axis=-1).astype(np.float32)
    # intensity 正規化
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6) * 255
    return vol, step_mm

def extract_bone_surface(volume, threshold_percentile=95, voxel_size=(0.2, 0.2, 0.5)):
    """
    volume: (H, W, D)
    voxel_size: (dy, dx, dz) [mm]
    """
    # 1. 閾値決定（高輝度＝骨）
    thr = np.percentile(volume, threshold_percentile)
    bone_mask = (volume > thr).astype(np.uint8)

    # 2. 3D形態学処理で骨面をなめらかに
    bone_bool = bone_mask.astype(bool)
    bone_bool = morphology.binary_closing(bone_bool, morphology.ball(1))
    bone_bool = morphology.remove_small_objects(bone_bool, min_size=500)

    # 3. Marching Cubesで表面抽出
    verts, faces, normals, _ = measure.marching_cubes(
        bone_bool, level=0.5, spacing=voxel_size
    )
    return verts, faces

def create_3d_figure(verts, faces):
    x, y, z = verts.T
    i, j, k = faces.T

    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='orange',
        opacity=1.0,
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3),
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig

# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(page_title="骨表面3Dエコー", layout="wide")
st.title("超音波エコーからの骨表面3D化デモ")

st.markdown("1. エコー動画（MP4）をアップロードすると、骨表面の3Dモデルをブラウザ上に表示します。")

uploaded_file = st.file_uploader("エコー動画（MP4）を選択してください", type=["mp4", "avi", "mov"])

col1, col2 = st.columns([1, 2])

with col1:
    thr_percent = st.slider("骨閾値パーセンタイル（感度）", 80, 99, 95)
    step_mm = st.number_input("フレーム間距離 [mm]", value=0.5, min_value=0.1, max_value=2.0, step=0.1)
    downsample = st.slider("フレーム間サンプリング（間引き）", 1, 5, 2,
                           help="例: 2なら1フレームおきに使用")

if uploaded_file is not None:
    # 一時ファイルとして保存
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    frames = []
    idx = 0
    st.info("動画を解析中…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % downsample != 0:
            idx += 1
            continue

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = preprocess_frame(gray)
        frames.append(processed)
        idx += 1

    cap.release()
    os.unlink(tfile.name)

    if len(frames) < 5:
        st.error("有効なフレームが少なすぎます。再撮影か、サンプリング設定を見直してください。")
    else:
        with st.spinner("3Dボリューム構築中…"):
            volume, step = frames_to_volume(frames, step_mm=step_mm)

        with st.spinner("骨表面抽出中…"):
            verts, faces = extract_bone_surface(volume, threshold_percentile=thr_percent,
                                                voxel_size=(0.2, 0.2, step))

        with col2:
            st.subheader("骨表面3D表示")
            fig = create_3d_figure(verts, faces)
            st.plotly_chart(fig, use_container_width=True)

        st.success(f"3Dメッシュ生成完了: 頂点数 {len(verts):,}, 面数 {len(faces):,}")

        # STLダウンロード（簡易ASCII）
        st.subheader("STLとしてダウンロード")
        st.caption("3Dプリンタや他ソフトで利用可能")

        def verts_faces_to_stl_ascii(verts, faces):
            buf = []
            buf.append("solid bone")
            for f in faces:
                v1, v2, v3 = verts[f]
                # 法線は簡易的に0 0 1
                buf.append(" facet normal 0 0 1")
                buf.append("  outer loop")
                buf.append(f"   vertex {v1[0]} {v1[1]} {v1[2]}")
                buf.append(f"   vertex {v2[0]} {v2[1]} {v2[2]}")
                buf.append(f"   vertex {v3[0]} {v3[1]} {v3[2]}")
                buf.append("  endloop")
                buf.append(" endfacet")
            buf.append("endsolid bone")
            return "\n".join(buf)

        stl_str = verts_faces_to_stl_ascii(verts, faces)
        st.download_button(
            "STLファイルをダウンロード",
            data=stl_str,
            file_name="bone_surface.stl",
            mime="application/sla"
        )
else:
    st.info("右上の「Browse files」からエコー動画をアップロードしてください。")
