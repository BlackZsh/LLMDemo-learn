import gradio as gra


def greet(name):
    return "Hello" + name + "!"


# text
# iface = gra.Interface(fn=greet, inputs="text", outputs="text")
iface = gra.Interface(fn=greet, inputs=gra.Textbox(lines=5, placeholder="plz input here"), outputs="text")
iface.launch()


# if __name__ == "__main__":
#     main()