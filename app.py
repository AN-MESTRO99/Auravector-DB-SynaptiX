# Tab 3: UMAP Topology & Nearest Neighbors Analysis
with tab_umap:
    st.markdown("##### High-Dimensional Vector Space Topology & Nearest Neighbors")
    st.caption("Maps query embedding proximity relative to corpus vectors via 2D UMAP projection and cosine metric distance.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    active_query = query if 'query' in locals() and query else "What is MS MARCO passage ranking?"
    
    if len(engine.doc_passages) > 0:
        from sklearn.metrics.pairwise import cosine_distances

        # 1. Prepare Query Vector
        query_vec = engine.vectorizer.transform([active_query]).toarray()
        if query_vec.shape[1] < 384:
            padding = np.zeros((1, 384 - query_vec.shape[1]))
            query_vec = np.hstack([query_vec, padding])
        else:
            query_vec = query_vec[:, :384]

        # 2. Select Corpus Subset for Projection
        corpus_limit = min(100, len(engine.doc_passages))
        corpus_vecs = engine.doc_vectors[:corpus_limit]
        
        # 3. Compute Exact Cosine Proximity to Query Vector
        dists = cosine_distances(query_vec, corpus_vecs)[0]
        sims = (1.0 - dists) * 100.0  # Percentage similarity
        
        # Identify Top 5 Nearest Neighbors
        nearest_indices = set(np.argsort(dists)[:5])

        # 4. UMAP Dimensionality Reduction
        all_vecs = np.vstack([query_vec, corpus_vecs])
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        projected = reducer.fit_transform(all_vecs)

        q_coords = projected[0]
        doc_coords = projected[1:]

        # 5. Build Structured Plot Data with Decreased Marker Sizes
        categories = []
        hover_texts = []
        sizes = []
        
        for idx in range(corpus_limit):
            doc_id = engine.doc_ids[idx]
            sim_score = sims[idx]
            text_snippet = engine.doc_passages[idx][:80] + "..."
            
            if idx in nearest_indices:
                categories.append("Nearest Neighbor")
                sizes.append(10)  # Decreased size for Nearest Neighbors
            else:
                categories.append("Unselected Corpus")
                sizes.append(5)   # Decreased size for background corpus dots
                
            hover_texts.append(f"<b>Doc ID:</b> {doc_id}<br><b>Similarity:</b> {sim_score:.2f}%<br><b>Passage:</b> {text_snippet}")

        # Add Query Vector Point
        x_pts = [q_coords[0]] + list(doc_coords[:, 0])
        y_pts = [q_coords[1]] + list(doc_coords[:, 1])
        all_categories = ["Query Vector"] + categories
        all_hovers = [f"<b>Search Query:</b> {active_query}"] + hover_texts
        all_sizes = [14] + sizes  # Decreased size for Query Vector dot

        # 6. Generate Plotly Figure with Blue Accent Color Map
        fig = px.scatter(
            x=x_pts,
            y=y_pts,
            color=all_categories,
            size=all_sizes,
            hover_name=all_hovers,
            color_discrete_map={
                "Query Vector": "#EF4444",        # Vibrant Red
                "Nearest Neighbor": "#2563EB",     # Vibrant Blue
                "Unselected Corpus": "#CBD5E1"    # Soft Slate Grey
            },
            labels={"x": "UMAP Axis 1", "y": "UMAP Axis 2", "color": "Vector Class"},
            template="plotly_white"
        )

        # Update Query Marker to Red Circle Dot
        fig.update_traces(
            selector=dict(name="Query Vector"),
            marker=dict(symbol="circle", color="#EF4444", line=dict(width=1.5, color="#991B1B"))
        )

        # Update Nearest Neighbors Markers to Blue Circle Dots
        fig.update_traces(
            selector=dict(name="Nearest Neighbor"),
            marker=dict(symbol="circle", color="#2563EB", line=dict(width=1.5, color="#1E40AF"))
        )

        # 7. Draw Visual Blue Connector Lines from Query to Nearest Neighbors
        for idx in nearest_indices:
            target_x = doc_coords[idx, 0]
            target_y = doc_coords[idx, 1]
            fig.add_shape(
                type="line",
                x0=q_coords[0], y0=q_coords[1],
                x1=target_x, y1=target_y,
                line=dict(color="#3B82F6", width=1.5, dash="dash"),
                layer="below"
            )

        # Layout Refinements
        fig.update_layout(
            height=540,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FAFC",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12, color="#334155")
            ),
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0", zeroline=False)
        )

        st.plotly_chart(fig, use_container_width=True)
