import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(train_loss_history, val_loss_history, num_epochs):

    fig = go.Figure()

    line_color = [px.colors.qualitative.Light24[5], px.colors.qualitative.Light24[4]]


    fig.add_trace(go.Scatter(x=list(range(1, num_epochs+1)), y=train_loss_history, mode='lines', line=dict(color=line_color[0], width=1), name="train loss"))
    fig.add_trace(go.Scatter(x=list(range(1, num_epochs+1)), y=val_loss_history, mode='lines',line=dict(color=line_color[1], width=1), name="valdiation loss")) #opacity=0.8

    fig.update_yaxes(range=[0, 1])
    fig.update_traces(textposition='top center')
    fig.update_layout(autosize=False,width=900, height=500, title_text="SNN Loss", title_x=0.5, xaxis_title="Epoch", yaxis_title="", xaxis = dict(tickmode='linear', tick0=1, dtick=1), legend=dict(yanchor="top",xanchor="right", x=1.35, y=1),template="plotly_dark")
    fig.show()


def plot_similarity_scores(non_matching_similarity, matching_similarity):

    train_num_batchs = len(non_matching_similarity)

    fig = go.Figure()

    line_color = [px.colors.qualitative.Light24[22], px.colors.qualitative.Light24[19]]


    fig.add_trace(go.Scatter(x=list(range(1, train_num_batchs+1)), y=non_matching_similarity, mode='lines', line=dict(color=line_color[0], width=1), name="unmatching categories"))
    fig.add_trace(go.Scatter(x=list(range(1, train_num_batchs+1)), y=matching_similarity, mode='lines',line=dict(color=line_color[1], width=1), name="matching categories")) #opacity=0.8

    fig.update_yaxes(range=[0, 1])
    fig.update_traces(textposition='top center')
    fig.update_layout(autosize=False,width=900, height=500, title_text="Similarity Scores", title_x=0.5, xaxis_title="Batch", yaxis_title="", legend=dict(yanchor="top",xanchor="right", x=1.35, y=1),template="plotly_dark")
    fig.show()



def show_space(X, title, colors=None, color_by="", show_3D=False):

  if show_3D:
    dictionary = dict(zip(pd.DataFrame(X).columns, ["COMP1", "COMP2", "COMP3"]))
    temp_df = pd.DataFrame(X).rename(columns=dictionary)
    fig = px.scatter_3d(temp_df, x='COMP1', y='COMP2', z='COMP3',color=colors, template="plotly_dark", labels={"color": color_by})
  else:
    dictionary = dict(zip(X.columns, ["COMP1", "COMP2"]))
    temp_df = pd.DataFrame(X).rename(columns=dictionary)
    fig = px.scatter(temp_df, x='COMP1', y='COMP2',color=colors, template="plotly_dark", labels={"color": color_by})

  fig.update_traces(marker=dict(size=4, opacity=0.98), textposition='top center')
  fig.update_layout(title_text=title, title_x=0.5, autosize=False,width=900, height=500, legend=dict(yanchor="top",xanchor="right", x=1.1, y=1))
  fig.show()
  #fig.write_html("file.html")



def plot_confusion_matrix(mat,fig_size, labels):

    fig = plt.figure(figsize=(fig_size,fig_size))
    ax= fig.add_subplot(1,1,1)
    sns.heatmap(mat, annot=True, cmap="Blues",ax = ax,fmt='g'); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
    plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()