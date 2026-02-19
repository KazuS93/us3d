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
FPSDEFAULT = 30

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


def extract_bone_surface(
    volume,
    threshold_percentile=82,
    voxel_x_mm=VOXEL_X_MM,
    voxel_y_mm=VOXEL_Y_MM,
    voxel_z_mm=0.5,
):
    """
    SNiBLE2 864x648向け 骨表面抽出（OpenCV+NumPy）
    voxel_x_mm, voxel_y_mm, voxel_z_mm: 1ピクセルあたりのmmスケール
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
    min_area = 10

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
        bone_mask = np.zeros_like(volume, dtype=np.uint8)
        for z in range(D):
            slice_any = np.zeros((H, W), dtype=np.uint8)
            for mask in candidates:
                slice_any = cv2.bitwise_or(slice_any, mask[:, :, z])
            kernel = np.ones((3, 3), np.uint8)
            slice_any = cv2.morphologyEx(slice_any, cv2.MORPH_OPEN, kernel)
            bone_mask[:, :, z] = slice_any

        pts = np.argwhere(bone_mask > 0)
        if pts.size == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)

    # ---- 5. 点群 → mm座標系へ変換（ここがキャリブレーション反映部分）----
    # pts: [z, y, x]
    zyx = pts
    verts = np.stack(
        [
            zyx[:, 2] * voxel_x_mm,  # x方向スケール
            zyx[:, 1] * voxel_y_mm,  # y方向スケール
            zyx[:, 0] * voxel_z_mm,  # z方向スケール
        ],
        axis=1,
    ).astype(np.float32)

    # ---- 6. 中央からの距離でノイズ除去 ----
    center = np.mean(verts, axis=0)
    dist = np.linalg.norm(verts - center, axis=1)
    med = np.median(dist)
    keep = dist < med * 2.0
    verts = verts[keep]

    # ---- 7. faces（STL用） ----
    n_faces = min(4000, max(len(verts) // 8, 0))
    if n_faces == 0:
        return verts, np.empty((0, 3), dtype=int)
    faces = np.random.randint(0, len(verts), size=(n_faces, 3), dtype=int)

    return verts, faces

# ==============================
# 3Dメッシュ描画（点群なし）
# ==============================

def create_3d_figure(verts, faces):
    """
    スマホ向け軽量表示:
      - Zが大きい点から骨表面メッシュを生成（ベージュ）
      - 点群表示なし
      - 表示範囲をやや広めにとる
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

    # 0. 負荷軽減のため上限5万点に間引き
    MAX_POINTS_FOR_SURFACE = 50000
    if len(verts) > MAX_POINTS_FOR_SURFACE:
        idx0 = np.random.choice(len(verts), MAX_POINTS_FOR_SURFACE, replace=False)
        x, y, z = x[idx0], y[idx0], z[idx0]

    # 1. Zが大きい点だけを表面候補に（上位20%）
    high_pct = 80.0
    z_thr = np.percentile(z, high_pct)
    mask_high = z >= z_thr
    xh, yh, zh = x[mask_high], y[mask_high], z[mask_high]

    # 2. XYグリッド化（40×40）で最大Zを高さに
    GRID_NX, GRID_NY = 40, 40
    x_min, x_max = xh.min(), xh.max()
    y_min, y_max = yh.min(), yh.max()

    xi = np.linspace(x_min, x_max, GRID_NX)
    yi = np.linspace(y_min, y_max, GRID_NY)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = np.full_like(Xi, np.nan, dtype=float)

    ix = np.clip(((xh - x_min) / (x_max - x_min + 1e-8) * (GRID_NX - 1)).astype(int), 0, GRID_NX - 1)
    iy = np.clip(((yh - y_min) / (y_max - y_min + 1e-8) * (GRID_NY - 1)).astype(int), 0, GRID_NY - 1)

    for gx, gy, gz in zip(ix, iy, zh):
        if np.isnan(Zi[gy, gx]) or gz > Zi[gy, gx]:
            Zi[gy, gx] = gz

    # 3. 大きく離れた高さを外れ値として除外
    vals = Zi[~np.isnan(Zi)]
    if vals.size > 0:
        med = np.median(vals)
        std = np.std(vals)
        tol = max(5.0, 2.0 * std)  # 5mm または 2σ以上外れをNaN
        Zi_clean = Zi.copy()
        bad = np.abs(Zi_clean - med) > tol
        Zi_clean[bad] = np.nan
    else:
        Zi_clean = Zi

    # 4. ベージュのメッシュだけ表示
    beige_color = "rgb(245, 222, 179)"
    surface = go.Surface(
        x=Xi,
        y=Yi,
        z=Zi_clean,
        colorscale=[[0, beige_color], [1, beige_color]],
        showscale=False,
        opacity=0.96,
        name="骨表面メッシュ",
    )

    fig = go.Figure(data=[surface])

    # 5. 軸レンジとカメラ（少し広め & 近め）
    x_pad = max(5.0, 0.2 * (x.max() - x.min()))
    y_pad = max(5.0, 0.2 * (y.max() - y.min()))
    z_pad = max(5.0, 0.2 * (z.max() - z.min()))

    x_range = [x.min() - x_pad, x.max() + x_pad]
    y_range = [y.min() - y_pad, y.max() + y_pad]
    z_range = [z.min() - z_pad, z.max() + z_pad]

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X [mm]", range=x_range),
            yaxis=dict(title="Y [mm]", range=y_range),
            zaxis=dict(title="Z [mm]", range=z_range),
            aspectmode="cube",
            camera=dict(eye=dict(x=0.9, y=0.9, z=1.1)),
        ),
        height=600,
        title="🦴 SNiBLE2 骨表面メッシュ（軽量）",
        showlegend=False,
    )
    return fig


# ==============================
# 一時ファイルユーティリティ
# ==============================

def write_bytes_to_tempfile(file_bytes, suffix=".mp4"):
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    return temp_path


# ==============================
# サムネイル生成（上10%＋右12.5%トリム）
# ==============================

def get_thumbnail_and_rois(file_bytes, grid_size=4,
                           top_trim_ratio=0.1, right_trim_ratio=0.1):
    """
    動画中央フレームからサムネイル生成。
    上側10%＋右側12.5%をトリミングしてから16分割。
    """
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

    # 上10%＋右12.5%をカット
    trim_top = int(h * top_trim_ratio)
    trim_right = int(w * right_trim_ratio)
    frame_trimmed = frame[trim_top:, : w - trim_right]

    h_trim, w_trim = frame_trimmed.shape[:2]

    thumbrgb = cv2.cvtColor(frame_trimmed, cv2.COLOR_BGR2RGB)

    tile_h, tile_w = h_trim // grid_size, w_trim // grid_size
    tiles = []
    coords_trimmed = []

    for gy in range(grid_size):
        for gx in range(grid_size):
            y1, y2 = gy * tile_h, (gy + 1) * tile_h
            x1, x2 = gx * tile_w, (gx + 1) * tile_w
            tile = thumbrgb[y1:y2, x1:x2]
            tiles.append(tile)
            coords_trimmed.append((x1, y1, x2, y2))

    # 元画像座標系に変換（Yはtrim_top分オフセット）
    coords_original = []
    for x1, y1, x2, y2 in coords_trimmed:
        coords_original.append((x1, y1 + trim_top, x2, y2 + trim_top))

    return thumbrgb, tiles, coords_original


# ==============================
# ROI付きフレーム前処理（上10%＋右12.5%トリム）
# ==============================

def load_and_preprocess_frames_roi(
    file_bytes,
    roi_indices,
    roi_coords,
    trim_sec=1.0,
    downsample=1,
    top_trim_ratio=0.1,
    right_trim_ratio=0.1,
):
    """
    上側10%＋右側12.5%トリム＋選択ROIでフレーム前処理
    """
    temp_path = write_bytes_to_tempfile(file_bytes, suffix=".mp4")
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        cap.release()
        os.remove(temp_path)
        raise RuntimeError("動画を開けませんでした")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPSDEFAULT
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # トリム用マスク（上10%＋右12.5%を0、それ以外1）
    trim_top = int(h * top_trim_ratio)
    trim_right = int(w * right_trim_ratio)
    mask_trim = np.zeros((h, w), dtype=np.uint8)
    mask_trim[trim_top:, : w - trim_right] = 1

    trim_frames = int(trim_sec * fps)
    start_frame = trim_frames
    end_frame = max(total_frames - trim_frames, start_frame + 10)

    # ROIが未選択なら全領域扱い
    if len(roi_indices) == 0:
        roi_indices = list(range(len(roi_coords)))

    # ROIマスク
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in roi_indices:
        x1, y1, x2, y2 = roi_coords[idx]
        roi_mask[y1:y2, x1:x2] = 1

    final_mask = roi_mask * mask_trim  # ROI ∩ トリム領域

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
        gray_masked = gray * final_mask

        ys, xs = np.where(final_mask > 0)
        if len(ys) == 0:
            frame_idx += 1
            continue
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        gray_roi = gray_masked[y_min : y_max + 1, x_min : x_max + 1]
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
st.title("🦴 SNiBLE2 ROI選択 骨表面3Dメッシュ（軽量版）")

st.markdown(
    """
**ワークフロー**

1. SNiBLE2で長軸方向にエコー動画を撮影（fps30, 1.5cm/s, 6〜8秒）
2. MP4動画をアップロード
3. 上側10%＋右側12.5%を自動トリミング → 16分割サムネイルから骨が写っているマスを選択
4. 「🚀 選択ROIで解析」で骨表面3Dメッシュを生成
"""
)

uploaded_file = st.file_uploader("📹 SNiBLE2 MP4動画をアップロード", type=["mp4"])

colleft, colright = st.columns([1, 2])

with colleft:
    thrpercent = st.slider("骨閾値", 75, 92, 82, 1, help="標準: 82")
    trimsec = st.slider("先頭/末尾トリム [秒]", 0.0, 2.0, 1.0, 0.1)
    downsample = st.slider("フレーム間引き", 1, 4, 1, help="1=高精度, 2=高速")

    # ★ タイル縦幅（mm）キャリブレーション
    tile_height_mm = st.slider("1タイル縦幅 [mm]", 6, 12, 8, 1,
                               help="16分割した1マスの実際の高さ [mm]")

if uploaded_file is not None:
    # ファイルバイトをセッションに保持
    if (st.session_state.file_name != uploaded_file.name) or (st.session_state.file_bytes is None):
        st.session_state.file_bytes = uploaded_file.getvalue()
        st.session_state.file_name = uploaded_file.name

    filebytes = st.session_state.file_bytes

    # サムネイル＆ROI生成
    with st.spinner("サムネイル生成中..."):
        thumbrgb, tiles, roicoords = get_thumbnail_and_rois(
            filebytes, grid_size=4, top_trim_ratio=0.1, right_trim_ratio=0.125
        )

    if thumbrgb is None:
        st.error("サムネイル生成に失敗しました（動画形式を確認）")
        st.stop()

    # ★ タイル高さ(px)から mm/px をキャリブレーション
    grid_size = 4
    tile_h_px = thumbrgb.shape[0] // grid_size
    mm_per_px = tile_height_mm / max(tile_h_px, 1)
    # Xも同じピッチとみなす
    voxel_x_mm_current = mm_per_px
    voxel_y_mm_current = mm_per_px

    with colleft:
        st.subheader("① トリミング済みサムネイル")
        st.image(thumbrgb, caption="中央フレーム（上10%＋右12.5%カット）", use_column_width=True)

    with colright:
        st.subheader("② 16分割ROI選択")
        selectedindices = []
        grid_size = 4
        for gy in range(grid_size):
            rowcols = st.columns(grid_size)
            for gx in range(grid_size):
                idx = gy * grid_size + gx
                tile = tiles[idx]
                with rowcols[gx]:
                    st.image(tile, use_column_width=True)
                    checked = st.checkbox(f"ROI {idx+1}", key=f"roi_{idx}")
                    if checked:
                        selectedindices.append(idx)

        st.markdown(f"**選択ROI: {len(selectedindices)} 個**（未選択なら全領域）")
        run_btn = st.button("🚀 選択ROIで解析")

    if run_btn:
        with st.spinner("③ 選択ROIでフレーム前処理中..."):
            frames = load_and_preprocess_frames_roi(
                filebytes,
                roi_indices=selectedindices,
                roi_coords=roicoords,
                trim_sec=trimsec,
                downsample=downsample,
                top_trim_ratio=0.1,
                right_trim_ratio=0.1,
            )

        if len(frames) < 10:
            st.error("有効フレームが少なすぎます。撮影時間を延ばすかトリム秒数を減らしてください。")
            st.stop()

        with st.spinner("④ 3Dボリューム構築＆骨抽出中..."):
            volume, _ = frames_to_volume(frames, step_mm=0.5)
            verts, faces = extract_bone_surface(
                volume,
                threshold_percentile=thrpercent,
                voxel_x_mm=voxel_x_mm_current,
                voxel_y_mm=voxel_y_mm_current,
                voxel_z_mm=0.5,  # ここは従来通り 0.5mm/フレーム
            )


        with colright:
            st.subheader("⑤ 骨表面3Dメッシュ")
            fig = create_3d_figure(verts, faces)
            st.plotly_chart(fig, use_container_width=True)

        st.success(f"✅ 完了: 頂点 {len(verts):,} 面 {len(faces):,}")

        # STL出力（ランダムfacesを利用）
        st.subheader("⑥ STLダウンロード（3Dプリント等）")

        def generate_stl(verts, faces, max_faces=4000):
            lines = ["solid SNiBLE2Bone"]
            n = min(max_faces, len(faces))
            for f in faces[:n]:
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
            lines.append("endsolid SNiBLE2Bone")
            return "\n".join(lines)

        stlcontent = generate_stl(verts, faces)
        st.download_button(
            "💾 STLをダウンロード",
            data=stlcontent,
            file_name=f"bone_roi_thr{thrpercent}_trim{trimsec:.1f}.stl",
            mime="application/octet-stream",
        )

        del frames, volume, verts, faces
        gc.collect()

else:
    st.info("📤 まずは SNiBLE2 の MP4 動画をアップロードしてください。")
    st.caption("推奨: fps30・1.5cm/s・6〜8秒の長軸スキャン")
