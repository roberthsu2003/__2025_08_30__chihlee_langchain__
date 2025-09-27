import gradio as gr
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# 初始化模型
model = OllamaLLM(model="gpt-oss:20b")

# 建立多變數的翻譯模板
complex_template = """
你是一位專業的{target_language}翻譯家，專精於{domain}領域。
請將以下{source_language}文本翻譯成{target_language}，並確保：
1. 保持原文的語氣和風格
2. 使用專業術語
3. 符合{target_language}的語言習慣

{source_language}文本：{text}
{target_language}翻譯：
"""

chat_prompt_template = ChatPromptTemplate.from_template(complex_template)

def translate_text(source_language, target_language, domain, text):
    """翻譯函數"""
    if not text.strip():
        return "請輸入要翻譯的文本"
    
    try:
        formatted_prompt = chat_prompt_template.format(
            target_language=target_language,
            source_language=source_language,
            domain=domain,
            text=text
        )
        
        response = model.invoke(formatted_prompt)
        return response
    except Exception as e:
        return f"翻譯過程中發生錯誤：{str(e)}"

# 建立Gradio介面
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="AI 專業翻譯助手",
    css="""
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: bold;
    }
    .header p {
        margin: 10px 0 0 0;
        font-size: 1.2em;
        opacity: 0.9;
    }
    """
) as demo:
    
    # 標題區域
    with gr.Row():
        gr.HTML("""
        <div class="header">
            <h1>🌐 AI 專業翻譯助手</h1>
            <p>使用先進的語言模型提供專業翻譯服務</p>
        </div>
        """)
    
    # 主要內容區域
    with gr.Row():
        with gr.Column(scale=1):
            # 輸入參數
            gr.Markdown("### ⚙️ 翻譯設定")
            
            source_lang = gr.Dropdown(
                choices=["英文", "日文", "韓文", "法文", "德文", "西班牙文", "義大利文", "俄文"],
                value="英文",
                label="源語言",
                info="選擇要翻譯的原始語言"
            )
            
            target_lang = gr.Dropdown(
                choices=["繁體中文", "簡體中文", "英文", "日文", "韓文", "法文", "德文", "西班牙文"],
                value="繁體中文",
                label="目標語言",
                info="選擇翻譯後的目標語言"
            )
            
            domain = gr.Dropdown(
                choices=["商業", "科技", "醫學", "法律", "文學", "學術", "日常", "新聞"],
                value="商業",
                label="專業領域",
                info="選擇文本所屬的專業領域"
            )
            
            # 輸入文本
            input_text = gr.Textbox(
                label="📝 輸入文本",
                placeholder="請輸入要翻譯的文本...",
                lines=8,
                max_lines=15,
                info="在此輸入需要翻譯的文本"
            )
            
            # 翻譯按鈕
            translate_btn = gr.Button(
                "🚀 開始翻譯",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            # 輸出區域
            gr.Markdown("### 📋 翻譯結果")
            
            output_text = gr.Textbox(
                label="翻譯結果",
                lines=8,
                max_lines=15,
                interactive=False,
                placeholder="翻譯結果將顯示在這裡..."
            )
            
            # 範例按鈕
            gr.Markdown("### 💡 快速範例")
            example_btn = gr.Button("載入範例文本", variant="secondary")
    
    # 事件處理
    translate_btn.click(
        fn=translate_text,
        inputs=[source_lang, target_lang, domain, input_text],
        outputs=output_text
    )
    
    example_btn.click(
        fn=lambda: "The quarterly revenue increased by 15% compared to last year.",
        outputs=input_text
    )
    
    # 頁腳
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤖 由 LangChain + Ollama 驅動 | 提供專業翻譯服務</p>
    </div>
    """)

# 啟動介面
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )