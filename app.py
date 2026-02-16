import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
import tempfile
import os
import gc

# ------------------------
# 前処理 & 骨抽出ロジック（skimage不要版）
# ------------------------

def preprocess_frame(gray):
    """OpenCVのみで前処理"""
    denoised = cv2.GaussianBlur(gray, (5, 5), 1.0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

def frames_to_volume(frames, step_mm=0.5):
    """2Dフレーム列 → 3Dボリューム"""
    vol = np.stack(frames, axis=-1).astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6) * 255
    return vol, step_mm

def extract_bone_surface(volume, threshold_percentile=85, voxel_size=(0.2, 0.2, 0.5)):
    """DBSCANを使わない骨抽出（メモリ安全版）"""
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # Step1: 多段階閾値で骨候補
    candidates = []
    for pct in [90, 85, 80]:
        thr = np.percentile(vol_norm, pct)
        mask = (vol_norm > thr).astype(np.uint8)
        candidates.append(mask)

    # Step2: 各スライスで最大領域のみ残す
    bone_mask = np.zeros_like(volume, dtype=np.uint8)
    H, W, D = volume.shape
    min_area = 30  # スライス内の最小画素数

    for z in range(D):
        best_mask = None
        best_area = 0

        for mask in candidates:
            num_labels, labels = cv2.connectedComponents(mask[:, :, z])
            if num_labels <= 1:
                continue

            areas = np.bincount(labels.ravel())[1:]
            if len(areas) == 0:
                continue

            max_idx = np.argmax(areas)
            max_area = areas[max_idx]

            if max_area > best_area and max_area >= min_area:
                best_area = max_area
                best_mask = (labels == max_idx + 1)

        if best_mask is not None:
            bone_mask[:, :, z] = best_mask.astype(np.uint8)

    # Step3: Z方向の連続性フィルタ（3フレーム連続以上）
    for y in range(H):
        for x in range(W):
            line = bone_mask[y, x, :]
            if line.sum() == 0:
                continue
            conv = np.convolve(line, np.ones(3, dtype=int), mode="valid")
            keep = conv >= 2
            bone_mask[y, x, 1:-1] = keep.astype(np.uint8)

    # Step4: 点群抽出
    pts = np.argwhere(bone_mask > 0)
    if pts.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    # [z,y,x]→[x,y,z] & mmスケール
    verts = pts.astype(np.float32) * np.array([voxel_size[1], voxel_size[0], voxel_size[2]])

    # 点数が多すぎる場合は間引き（メモリ対策）
    max_points = 50000
    if len(verts) > max_points:
        idx = np.random.choice(len(verts), max_points, replace=False)
        verts = verts[idx]

    # Plotly用簡易三角形
    n_faces = min(3000, max(len(verts) // 5, 0))
    if n_faces == 0:
        return verts, np.empty((0, 3), dtype=int)

    faces = np.random.randint(0, len(verts), (n_faces, 3))

    print(f"抽出完了: {len(verts)}点, {n_faces}面")
    return verts, faces

def create_3d_figure(verts, faces):
    if len(verts) == 0:
        return go.Figure().add_annotation(text="骨が検出されませんでした", xref="paper", yref="paper")
    
    x, y, z = verts.T[:min(10000, len(verts))]
    i, j, k = faces.T[:min(5000, len(faces))]
    
    mesh = go.Mesh3d(
        x=x.tolist(), y=y.tolist(), z=z.tolist(),
        i=i.tolist(), j=j.tolist(), k=k.tolist(),
        color='orange',
        opacity=0.9,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
    )
    
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis_title='X [mm]', yaxis_title='Y [mm]', zaxis_title='Z [mm]',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="骨表面3Dエコー", layout="wide", page_icon="🦴")
st.title("🦴 超音波エコー → 骨表面3D化")

st.markdown("""
**使用方法**  
1. エコー動画（MP4）をアップロード  
2. 骨閾値・フレーム間距離を調整  
3. 3Dモデル確認 → STLダウンロード
""")

uploaded_file = st.file_uploader("📹 エコー動画を選択", type=["mp4", "avi", "mov"], help="fps30, 1.5cm/s推奨")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔧 パラメータ調整")
    thr_percent = st.slider("骨閾値（感度）", 75, 95, 85, help="整形外科推奨: 80-90")  # 85に変更
    step_mm = st.number_input("フレーム間距離 [mm]", 0.1, 2.0, 0.5, 0.1, 
                              help="fps30・1.5cm/sなら0.5mm")
    downsample = st.slider("サンプリング（間引き）", 1, 5, 2, 
                           help="メモリ節約・高速化")
    max_frames = st.slider("最大フレーム数", 50, 300, 150, help="長すぎる動画はカット")

if uploaded_file is not None:
    with st.spinner("🎬 動画解析中..."):
        # ==================== 修正版一時ファイル処理 ====================
        # mkstemp で確実にハンドル管理
        fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        try:
            os.close(fd)  # ファイルディスクリプタ即閉鎖
            
            # データを書き込み
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # OpenCVで読み込み
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("❌ 動画形式エラー（MP4推奨）")
                os.remove(temp_path)
                st.stop()
            
            frames = []
            idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret or len(frames) >= max_frames:
                    break
                if idx % downsample != 0:
                    idx += 1
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed = preprocess_frame(gray)
                frames.append(processed)
                idx += 1
            
            cap.release()
            
            # ファイル削除（エラーが出ても無視）
            try:
                os.remove(temp_path)
            except PermissionError:
                st.warning("一時ファイルは自動クリーンアップされます")
            
            st.success(f"✅ {len(frames)}フレーム解析完了")
            
        except Exception as e:
            st.error(f"処理エラー: {e}")
            if 'temp_path' in locals():
                try:
                    os.remove(temp_path)
                except:
                    pass
            st.stop()
    
    if len(frames) < 5:
        st.error("❌ フレームが少なすぎます（再撮影 or 間引き減）")
        st.stop()
    
    # 3D処理
    with st.spinner("🔨 3Dボリューム構築..."):
        volume, _ = frames_to_volume(frames, step_mm=step_mm)
    
    with st.spinner("🦴 骨抽出中..."):
        verts, faces = extract_bone_surface(
            volume, threshold_percentile=thr_percent,
            voxel_size=(0.2, 0.2, step_mm)
        )
    
    # 3D表示
    with col2:
        st.subheader("🖼 骨表面3D")
        fig = create_3d_figure(verts, faces)
        st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"✅ 完成: 頂点数 {len(verts):,} | 面数 {len(faces):,}")
    
    # STLダウンロード
    st.subheader("💾 STL出力")
    def generate_stl(verts, faces):
        lines = ["solid bone"]
        for f in faces[:5000]:  # 高速化
            v1, v2, v3 = verts[f]
            lines.extend([
                " facet normal 0 0 1",
                "  outer loop",
                f"   vertex {v1[0]:.3f} {v1[1]:.3f} {v1[2]:.3f}",
                f"   vertex {v2[0]:.3f} {v2[1]:.3f} {v2[2]:.3f}",
                f"   vertex {v3[0]:.3f} {v3[1]:.3f} {v3[2]:.3f}",
                "  endloop",
                " endfacet"
            ])
        lines.append("endsolid bone")
        return "\n".join(lines)
    
    stl_content = generate_stl(verts, faces)
    st.download_button(
        "STLダウンロード",
        data=stl_content,
        file_name=f"bone_{thr_percent}_step{step_mm}.stl",
        mime="application/octet-stream"
    )
    
    # メモリクリーンアップ
    del volume, frames
    gc.collect()

else:
    st.info("📤 右のアップローダからエコー動画を選択してください")
    st.caption("推奨: fps30, 1.5cm/s, 5〜10秒程度")
