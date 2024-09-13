import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

#The Interface class is designed to create demos for machine learning models that accept one or more inputs and return one or more outputs.
demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"]
)
"""fn: The function to wrap a user interface (UI) around
inputs: The Gradio component(s) to use for the input. The number of components should match the number of arguments in your function.
outputs: The Gradio component(s) to use for the output. The number of components should match the number of return values from your function.
The fn argument is flexible — you can pass any Python function you want to wrap with a UI. In the example above, 
you saw a relatively simple function, but the function could be anything from a music generator to a tax calculator 
to the prediction function of a pretrained machine learning model."""


demo.launch()    