```python
import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(
    page_title="Live Road Detection",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Road Incident Detection")
st.caption("Video-analysis prototype | Uploaded footage only")

st.warning(
    "This prototype flags unusual visual motion for review. "
    "It does not confirm accidents or reliably predict severity."
)

video = st.file_uploader(
    "Upload road or CCTV footage",
    type=["mp4", "avi", "mov", "mkv"]
)

if video:
    st.video(video)

    if st.button("Analyze Video", type="primary"):
        path = None
        cap = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as f:
                f.write(video.getvalue())
                path = f.name

            cap = cv2.VideoCapture(path)

            if not cap.isOpened():
                st.error("Unable to open this video.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total // 300)
                previous = None
                events = []
                preview_frames = []
                frame_no = 0
                sampled = 0

                progress = st.progress(0)

                while cap.isOpened() and sampled < 300:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    current = frame_no
                    frame_no += 1

                    if current % step:
                        continue

                    gray = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2GRAY
                    )
                    gray = cv2.resize(gray, (320, 180))

                    if previous is not None:
                        diff = cv2.absdiff(previous, gray)
                        _, mask = cv2.threshold(
                            diff, 25, 255, cv2.THRESH_BINARY
                        )
                        motion = (
                            np.count_nonzero(mask) / mask.size
                        ) * 100

                        if motion >= 18:
                            events.append({
                                "Time (seconds)": round(
                                    current / fps, 2
                                ),
                                "Visual change (%)": round(
                                    motion, 1
                                ),
                                "Status": "Review footage"
                            })

                            if len(preview_frames) < 8:
                                preview = cv2.resize(
                                    frame, (480, 270)
                                )
                                preview = cv2.cvtColor(
                                    preview, cv2.COLOR_BGR2RGB
                                )

                                preview_frames.append({
                                    "Time (seconds)": round(
                                        current / fps, 2
                                    ),
                                    "Frame": preview
                                })

                    previous = gray
                    sampled += 1

                    if total:
                        progress.progress(
                            min(current / total, 1.0)
                        )

                cap.release()
                cap = None
                progress.progress(1.0)

                st.subheader("Analysis Results")
                st.metric("Frames sampled", sampled)
                st.metric("Motion flags", len(events))

                if events:
                    st.dataframe(
                        events,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("📸 Flagged Frame Previews")
                    st.caption(
                        "Sampled frames near detected visual changes. "
                        "These are not confirmed accidents."
                    )

                    for item in preview_frames:
                        st.markdown(
                            f"**Timestamp: {item['Time (seconds)']} seconds**"
                        )
                        st.image(item["Frame"], width=480)

                    st.warning(
                        "Motion flags are not confirmed accidents. "
                        "Review the footage before drawing conclusions."
                    )
                else:
                    st.info(
                        "No motion flags found at the selected threshold. "
                        "This does not prove there was no accident."
                    )

        except Exception as error:
            st.error(f"Video analysis failed: {error}")

        finally:
            if cap is not None:
                cap.release()
            if path and os.path.exists(path):
                os.remove(path)
```
