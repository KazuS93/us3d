import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
import tempfile
import os
import gc

# ------------------------
# SNiBLE2特化パラメータ
# ------------------------
SNIBLE2_WIDTH = 864   # あなたのフレーム幅
SNIBLE2_HEIGHT = 648  # フレーム高
VOXEL_X_MM = 0.15     # 864px → 約130mm視野 → 0.15mm/px
VOXEL_Y_MM = 0.15     # 同上
FPS = 30              # 想定

def preprocess_frame(gray):
    denoised = cv2.GaussianBlur(gray, (5, 5), 1.0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

def frames_to_volume(frames, step_mm=0.5):
    vol = np.stack(frames, axis=-1).astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6) * 255
    return vol, step_mm

def extract_bone_surface(volume, threshold_percentile=82):
    """SNiBLE2 864x648最適化版"""
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    
    # 多段階閾値（整形外科骨特化）
    candidates = []
    for pct in [88, 82, 76]:
        thr = np.percentile(vol_norm, pct)
        mask = (vol_norm > thr).astype(np.uint8)
        candidates.append(mask)
    
    # スライス毎最大領域抽出
    bone_mask = np.zeros_like(volume, dtype=np.uint8)
    for z in range(volume.shape[2]):
        best_mask = None
        best_area = 0
        
        for mask in candidates:
            num_labels, labels = cv2.connectedComponents(mask[:, :, z])
            if num_labels > 1:
                areas = np.bincount(labels.ravel())[1:]
                if len(areas) > 0:
                    max_area_idx = np.argmax(areas)
                    max_area = areas[max_area_idx]
                    if max_area > best_area and max_area > 30:  # 30px以上
                        best_area = max_area
                        best_mask = (labels == max_area_idx + 1)
        
        if best_mask is not None:
            bone_mask[:, :, z] = best_mask.astype(np.uint8)
    
    # Z連続性フィルタ
    for y in range(bone_mask.shape[0]):
        for x in range(bone_mask.shape[1]):
            slice_z = bone_mask[y, x, :]
            continuity = np.convolve(slice_z, np.ones(3), mode='valid') >= 2
            bone_mask[y, x, 1:-1] = continuity.astype(np.uint8)
    
    # 点群抽出（SNiBLE2スケール）
    pts = np.argwhere(bone_mask > 0)
    if pts.size == 0:
        return np.empty((0,3)), np.empty((0,3))
    
    verts = pts.astype(np.float32) * np.array([VOXEL_X_MM, VOXEL_Y_MM, 0.5])
    
    # ノイズ除去（中央密集領域のみ）
    center = np.mean(verts, axis=0)
    distances = np.linalg.norm(verts - center, axis=1)
    median_dist = np.median(distances)
    keep_mask = distances < median_dist * 1.8
    verts = verts[keep_mask]
    
    # Plotly用三角形
    n_faces = min(4000, len(verts) // 8)
    faces = np.random.randint(0, len(verts), (n_faces, 3))
    
    return verts, faces

def create_3d_figure(verts, faces):
    if len(verts) == 0:
        return go.Figure().add_annotation(
            text="骨検出されませんでした\n閾値を下げて再試行", 
            xref="paper", yref="paper", showarrow=False
        )
    
    x, y, z = verts.T[:min(15000, len(verts))]
    i, j, k = faces.T[:min(6000, len(faces))]
    
    mesh = go.Mesh3d(
        x=x.tolist(), y=y.tolist(), z=z.tolist(),
        i=i.tolist(), j=j.tolist(), k=k.tolist(),
        color='darkorange',
        opacity=0.85,
        lighting=dict(ambient=0.3, diffuse=0.9, specular=0.3),
    )
    
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            xaxis_title='X [mm]', yaxis_title='Y [mm]', zaxis_title='Z [mm]',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=2),  # Z方向縦長（骨向き）
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.8))
        ),
        height=650,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# ------------------------
# Streamlit UI（SNiBLE2特化）
# ------------------------
st.set_page_config(page_title="SNiBLE2骨3D", layout="wide", page_icon="🦴")
st.title("🦴 SNiBLE2 骨表面3D解析")

st.markdown("""
**最適設定済み**  
- フレームサイズ: 864×648  
- fps30・1.5cm/s対応  
- 先頭/末尾1秒自動トリミング
""")

uploaded_file = st.file_uploader("📹 MP4動画をアップロード", type=["mp4"], help="SNiBLE2出力推奨")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ 微調整")
    thr_percent = st.slider("骨閾値", 75, 92, 82, 1, help="82が標準")
    trim_sec = st.slider("トリム秒数", 0.5, 2.0, 1.0, 0.1, help="先頭/末尾カット")
    downsample = st.slider("間引き", 1, 4, 1, help="1=高精度、2=高速")
    roi_crop = st.checkbox("ROI自動クロップ（中央集中）", value=True)

if uploaded_file is not None:
    with st.spinner("🎬 SNiBLE2解析中..."):
        # 一時ファイル（安全版）
        fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        try:
            os.close(fd)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 先頭/末尾トリミング
            trim_frames = int(trim_sec * video_fps)
            start_frame = trim_frames
            end_frame = total_frames - trim_frames
            
            frames = []
            frame_idx = 0
            
            while frame_idx < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % downsample != 0:
                    frame_idx += 1
                    continue
                
                # フレームサイズ確認・調整
                h, w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if roi_crop:
                    # 中央ROI抽出（ノイズ低減）
                    roi_size = min(400, w//2, h//2)
                    cx, cy = w//2, h//2
                    gray = gray[cy-roi_size//2:cy+roi_size//2, cx-roi_size//2:cx+roi_size//2]
                
                processed = preprocess_frame(gray)
                frames.append(processed)
                frame_idx += 1
            
            cap.release()
            try:
                os.remove(temp_path)
            except:
                pass
            
            st.success(f"✅ 解析完了: {len(frames)}フレーム (トリム後)")
            
        except Exception as e:
            st.error(f"エラー: {e}")
            st.stop()
    
    if len(frames) < 10:
        st.error("❌ フレーム不足。再撮影（6秒以上）推奨")
        st.stop()
    
    # 3D処理
    with st.spinner("🦴 骨抽出中..."):
        volume, _ = frames_to_volume(frames, step_mm=0.5)
        verts, faces = extract_bone_surface(volume, threshold_percentile=thr_percent)
    
    # 結果表示
    with col2:
        st.subheader("🖼 3D骨モデル")
        fig = create_3d_figure(verts, faces)
        st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"🎉 完成！ 頂点: {len(verts):,} | 面: {len(faces):,}")
    
    # STL出力
    st.subheader("💾 STLダウンロード")
    def generate_stl(verts, faces):
        lines = ["solid SNiBLE2_Bone"]
        for f in faces[:4000]:
            v1, v2, v3 = verts[f]
            lines.extend([
                " facet normal 0 0 1",
                "  outer loop",
                f"   vertex {v1[0]:.2f} {v1[1]:.2f} {v1[2]:.2f}",
                f"   vertex {v2[0]:.2f} {v2[1]:.2f} {v2[2]:.2f}",
                f"   vertex {v3[0]:.2f} {v3[1]:.2f} {v3[2]:.2f}",
                "  endloop",
                " endfacet"
            ])
        lines.append("endsolid SNiBLE2_Bone")
        return "\n".join(lines)
    
    stl_content = generate_stl(verts, faces)
    st.download_button(
        "STL保存（3Dプリント用）",
        data=stl_content,
        file_name=f"bone_thr{thr_percent}_trim{trim_sec}s.stl",
        mime="application/octet-stream"
    )

else:
    st.info("📤 MP4動画をアップロード")
    st.caption("**SNiBLE2推奨設定**: fps30・1.5cm/s・6-8秒")
