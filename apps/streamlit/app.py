import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from fantasy_rag.embeddings.similarity_search import (
    SimilaritySearchEngine,
    SearchQuery,
)
from fantasy_rag.embeddings.clip_embedder import CLIPConfig
from fantasy_rag.config.settings import RENDERS_DIR, DATABASE_DIR, DATA_DIR, GENERATED_DIR
from fantasy_rag.generation.shape_generator import ShapeGenerator, ShapeConfig, HY3D_AVAILABLE
from fantasy_rag.generation.background import BG_MODELS

CLIP_SIM_FLOOR = 0.05
CLIP_SIM_CEIL = 0.35


def rescale_score(raw_cosine: float) -> float:
    clamped = max(CLIP_SIM_FLOOR, min(CLIP_SIM_CEIL, raw_cosine))
    return (clamped - CLIP_SIM_FLOOR) / (CLIP_SIM_CEIL - CLIP_SIM_FLOOR) * 100


@st.cache_resource
def load_search_engine():
    engine = SimilaritySearchEngine(
        db_path=str(DATABASE_DIR),
        clip_config=CLIPConfig(),
    )
    engine.initialize()
    return engine


def get_shape_generator(**options) -> ShapeGenerator:
    if "shape_generator" not in st.session_state:
        st.session_state.shape_generator = None

    gen = st.session_state.shape_generator
    if gen is None:
        gen = ShapeGenerator(ShapeConfig(**options))
        st.session_state.shape_generator = gen
    else:
        for key, value in options.items():
            setattr(gen.config, key, value)
    return gen


def find_asset_thumbnail(asset_id: str, metadata: dict) -> Path | None:
    iso_path = RENDERS_DIR / asset_id / f"{asset_id}_iso_3d.png"
    if iso_path.exists():
        return iso_path

    filename = metadata.get("original_filename", "")
    if filename:
        for search_dir in [DATA_DIR, RENDERS_DIR, PROJECT_ROOT / "data"]:
            candidate = search_dir / filename
            if candidate.exists():
                return candidate

    return None


def show_asset_views(asset_id: str, metadata: dict):
    views = ["front", "back", "left", "right", "top", "iso_3d"]
    render_dir = RENDERS_DIR / asset_id
    found = []
    for view in views:
        path = render_dir / f"{asset_id}_{view}.png"
        if path.exists():
            found.append((view, path))

    if found:
        cols = st.columns(3)
        for i, (view, path) in enumerate(found):
            cols[i % 3].image(str(path), caption=view, use_container_width=True)
    else:
        thumb = find_asset_thumbnail(asset_id, metadata)
        if thumb:
            st.image(str(thumb), caption="source image", use_container_width=True)
        else:
            st.info("No rendered views available.")


# --- Search Tab ---

def tab_search():
    engine = load_search_engine()

    col_search, col_opts = st.columns([3, 1])
    with col_search:
        query = st.text_input(
            "Search",
            placeholder="magical sword, dragon shield, enchanted armor...",
        )
    with col_opts:
        max_results = st.slider("Max results", 1, 20, 5)

    if not query:
        stats = engine.get_collection_stats()
        n = stats.get("total_documents", 0)
        if n > 0:
            st.info(f"{n} assets indexed. Type a query to search.")
        else:
            st.warning("No assets indexed yet. Run the indexing pipeline first.")
        return

    search_query = SearchQuery(text=query, max_results=max_results)
    response = engine.search(search_query)

    st.write(f"**{response.total_found}** results in {response.processing_time:.3f}s")

    for result in response.results:
        thumb = find_asset_thumbnail(result.asset_id, result.metadata)

        col_img, col_info = st.columns([1, 2])
        with col_img:
            if thumb:
                st.image(str(thumb), width=280)
            else:
                st.write(f"`{result.asset_id}`")

        with col_info:
            filename = result.metadata.get("original_filename", result.asset_id)
            st.subheader(filename)
            pct = rescale_score(result.similarity_score)
            st.metric("Match", f"{pct:.0f}%", help=f"Raw CLIP cosine: {result.similarity_score:.4f}")

            with st.expander("All views"):
                show_asset_views(result.asset_id, result.metadata)

        st.divider()


# --- Generate Tab ---

def tab_generate():
    if not HY3D_AVAILABLE:
        st.error("Hunyuan3D is not installed. Clone and install from third_party/Hunyuan3D-2.1.")
        return

    uploaded = st.file_uploader(
        "Upload a concept image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if not uploaded:
        st.info("Upload an image to generate a 3D mesh.")
        return

    col_preview, col_options = st.columns([1, 1])
    with col_preview:
        st.image(uploaded, caption=uploaded.name, width=400)

    with col_options:
        export_format = st.selectbox("Export format", ["glb", "stl", "obj", "ply"])
        steps = st.slider("Inference steps", 10, 100, 50,
                          help="More steps = better quality, slower")
        remove_bg = st.checkbox("Remove background", value=True,
                                help="Remove image background before generation (recommended)")
        bg_model = st.selectbox(
            "Background model", BG_MODELS, disabled=not remove_bg,
            help="birefnet-general: best on fine details. isnet-anime: fast, good on illustrations. "
                 "u2net: legacy, soft halos that break reconstruction",
        )
        refine = st.checkbox("Refine mask (solid hair / thin parts)", value=True, disabled=not remove_bg,
                             help="Binarize the mask and thicken thin strands so they become solid geometry")
        remove_floaters = st.checkbox("Remove floating parts", value=True,
                                      help="Drop small disconnected mesh pieces after generation")
        seed_text = st.text_input("Seed (empty = random)", value="",
                                  help="Fix the seed to compare settings on the same image")

    col_gen, col_unload = st.columns([1, 1])
    with col_gen:
        generate_clicked = st.button("Generate 3D Mesh", type="primary")
    with col_unload:
        if st.button("Unload model (free VRAM)"):
            if "shape_generator" in st.session_state and st.session_state.shape_generator:
                st.session_state.shape_generator.unload()
                st.session_state.shape_generator = None
            st.success("Model unloaded, VRAM freed.")

    if generate_clicked:
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_image = tmp_dir / f"input{Path(uploaded.name).suffix}"
        tmp_image.write_bytes(uploaded.getvalue())

        with st.status("Generating 3D mesh...", expanded=True) as status:
            st.write("Loading model (first time: ~2-3 min)...")
            seed = int(seed_text) if seed_text.strip().lstrip("-").isdigit() else None
            generator = get_shape_generator(
                num_inference_steps=steps,
                remove_background=remove_bg,
                bg_model=bg_model,
                refine_mask=refine,
                remove_floaters=remove_floaters,
                seed=seed,
            )

            if remove_bg:
                st.write(f"Removing background ({bg_model})...")

            st.write(f"Generating shape ({steps} steps, ~2-4 min)...")
            result = generator.generate(tmp_image)

            if not result.success:
                status.update(label="Generation failed", state="error")
                st.error(result.error_message)
                return

            stem = Path(uploaded.name).stem
            output_dir = GENERATED_DIR / stem
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{stem}.{export_format}"
            generator.export_mesh(result.mesh, output_file, fmt=export_format)

            status.update(label=f"Done in {result.generation_time:.0f}s", state="complete")

        col_stats, col_dl = st.columns([1, 1])
        with col_stats:
            st.metric("Vertices", f"{result.vertices_count:,}")
            st.metric("Faces", f"{result.faces_count:,}")

        with col_dl:
            with open(output_file, "rb") as f:
                st.download_button(
                    f"Download .{export_format}",
                    data=f.read(),
                    file_name=output_file.name,
                    mime="application/octet-stream",
                )

        st.caption(f"Saved to: `{output_file}`")


# --- Main ---

def main():
    st.set_page_config(page_title="Fantasy RAG", layout="wide")
    st.title("Fantasy RAG")

    tab1, tab2 = st.tabs(["Search", "Generate 3D"])

    with tab1:
        tab_search()

    with tab2:
        tab_generate()


if __name__ == "__main__":
    main()
