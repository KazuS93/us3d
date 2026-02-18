import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
import tempfile
import os
import gc

# ==============================
# パラメータ（SNiBLE2想定）
# ==============================
SNIBLE2_WIDTH = 864
SNIBLE2_HEIGHT = 648
VOXEL_X_MM = 0.15   # 約0.15mm/px（視野 ~130mm想定）
VOXEL_Y_MM = 0.15
FPS_DEFAULT = 30

# セッション状態初期化
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes = None
    st.session_state.file_name = None

# ==============================
# 前処理 & 骨抽出ロジック
# ==============================

def preprocess_frame(gray):
    """エコー画像前処理（OpenCVのみ）"""
    denoised = cv2.GaussianBlur(gray, (5, 5), 1.0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

def frames_to_volume(frames, step_mm=0.5):
    """2Dフレーム列 → 3Dボリューム"""
    vol = np.stack(frames, axis=-1).astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6) * 255
    return vol, step_mm

def extract_bone_surface(volume, threshold_percentile=82):
    """
    SNiBLE2 864x648向け 骨表面抽出（OpenCV+NumPy）
    threshold_percentile: スライダーの値（例: 75〜92）をそのまま使う
    """
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    H, W, D = volume.shape

    # ---- 1. スライダー値を反映した多段階閾値 ----
    base = float(threshold_percentile)
    cand_perc = sorted(set([
        max(50.0, min(99.0, base + 6)),
        max(50.0, min(99.0, base)),
        max(50.0, min(99.0, base - 6)),
    ]))
    candidates = []
    for pct in cand_perc:
        thr = np.percentile(vol_norm, pct)
        mask = (vol_norm > thr).astype(np.uint8)
        candidates.append(mask)

    # ---- 2. スライス毎 最大領域を採用（厳しめマスク）----
    bone_mask_strict = np.zeros_like(volume, dtype=np.uint8)
    min_area = 10  # 面積しきい値を少し緩めに

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
            max_area_idx = np.argmax(areas)
            max_area = areas[max_area_idx]
            if max_area > best_area and max_area >= min_area:
                best_area = max_area
                best_mask = (labels == (max_area_idx + 1))
        if best_mask is not None:
            bone_mask_strict[:, :, z] = best_mask.astype(np.uint8)

    # ---- 3. Z方向連続性フィルタ（3フレーム中2以上）----
    bone_mask = bone_mask_strict.copy()
    if D >= 3:
        for y in range(H):
            for x in range(W):
                line = bone_mask_strict[y, x, :]
                cont = np.convolve(line, np.ones(3, dtype=int), mode="valid") >= 2
                bone_mask[y, x, 1:-1] = cont.astype(np.uint8)

    pts = np.argwhere(bone_mask > 0)

    # ---- 4. 厳しすぎて0点なら「ゆるいマスク」にフォールバック ----
    if pts.size == 0:
        # 連続性フィルタを外し、最大領域＋候補閾値だけで判定
        bone_mask = np.zeros_like(volume, dtype=np.uint8)
        for z in range(D):
            slice_any = np.zeros((H, W), dtype=np.uint8)
            for mask in candidates:
                slice_any = cv2.bitwise_or(slice_any, mask[:, :, z])
            # 小さなノイズ除去（2Dモルフォロジ）
            kernel = np.ones((3, 3), np.uint8)
            slice_any = cv2.morphologyEx(slice_any, cv2.MORPH_OPEN, kernel)
            bone_mask[:, :, z] = slice_any

        pts = np.argwhere(bone_mask > 0)
        if pts.size == 0:
            # 本当に何もない場合は空で返す
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)

    # ---- 5. 点群 → mm座標系へ変換 ----
    zyx = pts  # [z, y, x]
    verts = np.stack(
        [
            zyx[:, 2] * VOXEL_X_MM,  # x
            zyx[:, 1] * VOXEL_Y_MM,  # y
            zyx[:, 0] * 0.5,         # z: 0.5mm/フレーム
        ],
        axis=1,
    ).astype(np.float32)

    # ---- 6. 中央からの距離で緩いノイズ除去 ----
    center = np.mean(verts, axis=0)
    dist = np.linalg.norm(verts - center, axis=1)
    med = np.median(dist)
    keep = dist < med * 2.0  # 1.8 → 2.0 に緩和
    verts = verts[keep]

    # ---- 7. プロット用の三角形生成 ----
    n_faces = min(4000, max(len(verts) // 8, 0))
    if n_faces == 0:
        return verts, np.empty((0, 3), dtype=int)

    faces = np.random.randint(0, len(verts), size=(n_faces, 3), dtype=int)

    return verts, faces

def create_3d_figure(verts, faces):
    """
    骨点群から:
      1) Zが大きい点だけ取り出し（骨表面想定）
      2) XYをグリッド化し、各セルの最大Zを高さとする
      3) 大きく外れた高さを除外
      4) ベージュ単色の3Dメッシュ + 元の点群を表示
    """
    if len(verts) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="骨が検出されませんでした",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    verts = np.asarray(verts, dtype=float)
    x, y, z = verts.T

    # ---------- 1. Zが大きい点だけを採用（上位20%） ----------
    high_pct = 80.0   # 上位20%を骨表面候補に
    z_thr = np.percentile(z, high_pct)
    mask_high = z >= z_thr
    xh, yh, zh = x[mask_high], y[mask_high], z[mask_high]

    # ---------- 2. XYグリッド化して最大Zを高さに ----------
    grid_nx = 80   # グリッド分割数（必要なら調整）
    grid_ny = 80

    x_min, x_max = xh.min(), xh.max()
    y_min, y_max = yh.min(), yh.max()

    xi = np.linspace(x_min, x_max, grid_nx)
    yi = np.linspace(y_min, y_max, grid_ny)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = np.full_like(Xi, np.nan, dtype=float)

    # 各点をグリッドに割り当て、同じセルなら最大Zを残す
    ix = np.clip(((xh - x_min) / (x_max - x_min + 1e-8) * (grid_nx - 1)).astype(int), 0, grid_nx - 1)
    iy = np.clip(((yh - y_min) / (y_max - y_min + 1e-8) * (grid_ny - 1)).astype(int), 0, grid_ny - 1)

    for px, py, pz in zip(ix, iy, zh):
        if np.isnan(Zi[py, px]) or pz > Zi[py, px]:
            Zi[py, px] = pz

    # ---------- 3. 大きく離れている高さを外れ値として除外 ----------
    vals = Zi[~np.isnan(Zi)]
    if vals.size > 0:
        med = np.median(vals)
        std = np.std(vals)
        tol = max(5.0, 2.0 * std)  # 2σ または5mm以上離れているセルを除外
        Zi_clean = Zi.copy()
        mask_outlier = np.abs(Zi_clean - med) > tol
        Zi_clean[mask_outlier] = np.nan
    else:
        Zi_clean = Zi

    # ---------- 4. 3Dメッシュ（ベージュ） + 元点群を表示 ----------
    # メッシュ：ベージュ単色
    beige_color = "rgb(245, 222, 179)"  # wheat / beige

    surface = go.Surface(
        x=Xi,
        y=Yi,
        z=Zi_clean,
        colorscale=[[0, beige_color], [1, beige_color]],
        showscale=False,
        opacity=0.95,
        name="骨表面メッシュ",
    )

    # 元の点群（低Zも含めて奥行き確認用）
    scatter = go.Scatter3d(
        x=x.tolist(),
        y=y.tolist(),
        z=z.tolist(),
        mode="markers",
        marker=dict(
            size=1.5,
            color=z,
            colorscale="Viridis",
            opacity=0.35,
            colorbar=dict(title="Z [mm]", x=1.02),
        ),
        name="骨点群",
        showlegend=True,
    )

    fig = go.Figure(data=[surface, scatter])

    # 軸レンジ自動設定
    x_range = [x.min() - 2, x.max() + 2]
    y_range = [y.min() - 2, y.max() + 2]
    z_range = [z.min() - 2, z.max() + 2]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X [mm]", range=x_range),
            yaxis=dict(title="Y [mm]", range=y_range),
            zaxis=dict(title="Z [mm]", range=z_range),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.6)),
        ),
        height=700,
        title="🦴 SNiBLE2 骨表面メッシュ＋点群",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )
    return fig

# ==============================
# ヘルパー：動画→テンポラリ保存
# ==============================

def write_bytes_to_tempfile(file_bytes, suffix=".mp4"):
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    return temp_path

# ==============================
# ① サムネイル生成 & 16分割
# ==============================

def get_thumbnail_and_rois(file_bytes, grid_size=4, top_trim_ratio=0.1):
    """動画中央フレームをサムネイル化。上側10%カットして16分割"""
    temp_path = write_bytes_to_tempfile(file_bytes, suffix=".mp4")
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        cap.release()
        os.remove(temp_path)
        return None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret, frame = cap.read()
    cap.release()
    os.remove(temp_path)

    if not ret:
        return None, None, None

    h, w = frame.shape[:2]
    
    # 上側10%トリミング
    trim_top = int(h * top_trim_ratio)
    frame_trimmed = frame[trim_top:, :]  # 上から10%カット
    h_trim, w_trim = frame_trimmed.shape[:2]
    
    # RGB変換
    thumb_rgb = cv2.cvtColor(frame_trimmed, cv2.COLOR_BGR2RGB)

    tile_h, tile_w = h_trim // grid_size, w_trim // grid_size
    tiles = []
    coords_trimmed = []  # トリミング後の座標

    for gy in range(grid_size):
        for gx in range(grid_size):
            y1, y2 = gy * tile_h, (gy + 1) * tile_h
            x1, x2 = gx * tile_w, (gx + 1) * tile_w
            tile = thumb_rgb[y1:y2, x1:x2]
            tiles.append(tile)
            coords_trimmed.append((x1, y1, x2, y2))  # トリミング後座標

    # 元画像座標に変換（上側10%分オフセット）
    coords_original = []
    for x1, y1, x2, y2 in coords_trimmed:
        coords_original.append((x1, y1 + trim_top, x2, y2 + trim_top))

    return thumb_rgb, tiles, coords_original

# ==============================
# ② 選択ROIのみでフレーム前処理
# ==============================

def load_and_preprocess_frames_roi(file_bytes, roi_indices, roi_coords,
                                   trim_sec=1.0, downsample=1, top_trim_ratio=0.1):
    """
    上側10%カット＋選択ROIでフレーム前処理
    """
    temp_path = write_bytes_to_tempfile(file_bytes, suffix=".mp4")
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        cap.release()
        os.remove(temp_path)
        raise RuntimeError("動画を開けませんでした")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 上側10%トリミング用マスク作成
    trim_top = int(h * top_trim_ratio)
    mask_trim = np.zeros((h, w), dtype=np.uint8)
    mask_trim[trim_top:, :] = 1  # 上側10%以外を有効

    trim_frames = int(trim_sec * fps)
    start_frame = trim_frames
    end_frame = max(total_frames - trim_frames, start_frame + 10)

    # ROI選択（未選択なら全領域）
    if len(roi_indices) == 0:
        roi_indices = list(range(len(roi_coords)))

    # ROI＋上側トリムマスク
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in roi_indices:
        x1, y1, x2, y2 = roi_coords[idx]
        roi_mask[y1:y2, x1:x2] = 1

    final_mask = roi_mask * mask_trim  # ROI ∩ 上側トリム

    frames = []
    frame_idx = 0

    while True:
        pos = start_frame + frame_idx
        if pos >= end_frame:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % downsample != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_masked = gray * final_mask  # ROI＋上側トリム適用

        # 有効領域のみ抽出（バウンディングボックス）
        ys, xs = np.where(final_mask > 0)
        if len(ys) == 0:
            frame_idx += 1
            continue
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        gray_roi = gray_masked[y_min:y_max+1, x_min:x_max+1]

        processed = preprocess_frame(gray_roi)
        frames.append(processed)
        frame_idx += 1

    cap.release()
    os.remove(temp_path)
    return frames

# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="SNiBLE2 ROI骨3D", layout="wide", page_icon="🦴")
st.title("🦴 SNiBLE2 超音波エコー ROI選択 骨表面3D解析")

st.markdown(
    """
**ワークフロー**

1. SNiBLE2で長軸方向にエコー動画を撮影（fps30, 1.5cm/s, 6〜8秒）
2. MP4動画をアップロード
3. サムネイルを16分割 → 骨が写っているマスを複数選択
4. 「選択ROIで解析」ボタンで3D骨モデルを生成
"""
)

uploaded_file = st.file_uploader("📹 SNiBLE2 MP4動画をアップロード", type=["mp4"])

col_left, col_right = st.columns([1, 2])

with col_left:
    thr_percent = st.slider("骨閾値", 75, 92, 82, 1, help="標準: 82")
    trim_sec = st.slider("先頭/末尾トリム [秒]", 0.0, 2.0, 1.0, 0.1)
    downsample = st.slider("フレーム間引き", 1, 4, 1, help="1=高精度, 2=高速")

if uploaded_file is not None:
    # ファイルバイトをセッションに保持
    if (st.session_state.file_name != uploaded_file.name) or (st.session_state.file_bytes is None):
        st.session_state.file_bytes = uploaded_file.getvalue()
        st.session_state.file_name = uploaded_file.name

    file_bytes = st.session_state.file_bytes

    # サムネイル＆16分割ROI生成
    with st.spinner("サムネイル生成中..."):
        thumb_rgb, tiles, roi_coords = get_thumbnail_and_rois(file_bytes, grid_size=4)

    if thumb_rgb is None:
        st.error("サムネイル生成に失敗しました（動画形式を確認）")
        st.stop()

    with col_left:
        st.subheader("① 動画サムネイル")
        st.image(thumb_rgb, caption="中央フレーム", use_column_width=True)

    with col_right:
        st.subheader("② 16分割ROI選択（上側10%トリミング後）")
        st.caption("上側10%は自動でカットされています。骨が写っているマスを選択してください。")

        selectedindices = []
        grid_size = 4

        for gy in range(grid_size):
            row_cols = st.columns(grid_size)
            for gx in range(grid_size):
                idx = gy * grid_size + gx
                tile = tiles[idx]
                with row_cols[gx]:
                    st.image(tile, use_column_width=True)
                    checked = st.checkbox(f"ROI {idx+1}", key=f"roi_{idx}")
                    if checked:
                        selectedindices.append(idx)

    st.markdown(f"**選択ROI: {len(selectedindices)} 個**（未選択時は全領域解析）")

    run_btn = st.button("🚀 選択ROIで解析")



    if run_btn:
        with st.spinner("③ 選択ROIでフレーム前処理中..."):
            frames = load_and_preprocess_frames_roi(
                file_bytes,
                roi_indices=selectedindices,
                roi_coords=roi_coords,
                trim_sec=trim_sec,
                downsample=downsample,
            )

        if len(frames) < 10:
            st.error("有効フレームが少なすぎます。撮影時間を延ばすかトリム秒数を減らしてください。")
            st.stop()

        with st.spinner("④ 3Dボリューム構築＆骨抽出中..."):
            volume, _ = frames_to_volume(frames, step_mm=0.5)
            verts, faces = extract_bone_surface(volume, threshold_percentile=thr_percent)

        with col_right:
            st.subheader("⑤ 3D骨モデル")
            fig = create_3d_figure(verts, faces)
            st.plotly_chart(fig, use_container_width=True)

        st.success(f"✅ 完了: 頂点 {len(verts):,} 面 {len(faces):,}")

        # STL出力
        st.subheader("⑥ STLダウンロード（3Dプリント等）")

        def generate_stl(verts, faces):
            lines = ["solid SNiBLE2_Bone"]
            max_faces = min(4000, len(faces))
            for f in faces[:max_faces]:
                v1, v2, v3 = verts[f]
                lines.extend(
                    [
                        " facet normal 0 0 1",
                        "  outer loop",
                        f"   vertex {v1[0]:.2f} {v1[1]:.2f} {v1[2]:.2f}",
                        f"   vertex {v2[0]:.2f} {v2[1]:.2f} {v2[2]:.2f}",
                        f"   vertex {v3[0]:.2f} {v3[1]:.2f} {v3[2]:.2f}",
                        "  endloop",
                        " endfacet",
                    ]
                )
            lines.append("endsolid SNiBLE2_Bone")
            return "\n".join(lines)

        stl_content = generate_stl(verts, faces)
        st.download_button(
            "💾 STLをダウンロード",
            data=stl_content,
            file_name=f"bone_roi_thr{thr_percent}_trim{trim_sec:.1f}.stl",
            mime="application/octet-stream",
        )

        # メモリ解放
        del frames, volume, verts, faces
        gc.collect()
else:
    st.info("📤 まずは SNiBLE2 の MP4 動画をアップロードしてください。")
    st.caption("推奨: fps30・1.5cm/s・6〜8秒の長軸スキャン")
