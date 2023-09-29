import streamlit as st
import retriever_service
import reader_service

COLBERT = "ColBERT"
MDR = "Multi-Hop Dense Text Retrieval (MDR)"


@st.cache_resource
def load_colbert_comps():
    searcher, collections = retriever_service.get_colbert_components()
    return searcher, collections


@st.cache_resource
def load_mdr_comps():
    retriever, index, id2doc, tokenizer, args = retriever_service.get_mdr_components()
    return retriever, index, id2doc, tokenizer, args


@st.cache_resource
def load_reader():
    question_answerer = reader_service.init_qa()
    return question_answerer


st.title("Vietnamese QA Demo", anchor=None)
chosen_retriever = st.selectbox("Please choose a Retriever method", (COLBERT, MDR))
top_k = st.number_input(
    "Select number of top passages", min_value=1, max_value=100, step=1, value=10
)

if chosen_retriever == COLBERT:
    sample_query = "Vị trí địa lý của Paris có gì đặc biệt?"
else:
    sample_query = (
        "Câu lạc bộ nào James Milner từng chơi có trụ sở tại Swindon, Wiltshire, Anh?"
    )
query = st.text_input("Enter your question", sample_query)

if st.button("Get Answer", type="primary"):
    with st.spinner(text="Processing..."):
        if chosen_retriever == COLBERT:
            colbert_searcher, colbert_collections = load_colbert_comps()
            retr_results = retriever_service.colbert_search(
                colbert_searcher, colbert_collections, query, top_k
            )
        else:
            retriever, index, id2doc, tokenizer, args = load_mdr_comps()
            retr_results = retriever_service.mdr_search(
                retriever, index, id2doc, tokenizer, args, query, top_k
            )

        question_answerer = load_reader()
        if chosen_retriever == COLBERT:
            preds, contexts = reader_service.get_qa_preds(
                question_answerer, query, retr_results.at[0, "passage"]
            )
        else:
            c = retr_results.at[0, "passage1"] + " " + retr_results.at[0, "passage2"]
            preds, contexts = reader_service.get_qa_preds(question_answerer, query, c)
        st.header("Answer")
        st.info(preds["answer"])
        st.caption(f"Score: {preds['score']}")

        st.header("Retrieved passages")
        if chosen_retriever == COLBERT:
            st.dataframe(
                retr_results,
                column_config={
                    "passage": st.column_config.TextColumn(
                        "Retrieved Passages (Double click to see full passage)",
                        help="Double click to see full passage",
                        max_chars=2000,
                    ),
                    "score": "Score",
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.dataframe(
                retr_results,
                column_config={
                    "passage1": st.column_config.TextColumn(
                        "Retrieved Passages 1 (Double click to see full passage)",
                        help="Double click to see full passage",
                    ),
                    "passage2": st.column_config.TextColumn(
                        "Retrieved Passages 2 (Double click to see full passage)",
                        help="Double click to see full passage",
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )
