import altair as alt

def embedding_plot(mdm_df, emb):
    emb_options = {
        'RSA': 'mdm', 
        'CKA': 'cka'
        }
    assert emb in emb_options.keys(), f"Chosen emb: {emb} not in options {list(emb_options.keys())}"

    slider = alt.binding_range(min=0, max=100, step=1)
    select_epoch = alt.selection_single(name="epoch", fields=['epoch'],
                                    bind=slider, init={'epoch': 0})

    # A dropdown filter
    data_name_dropdown = alt.binding_select(options=mdm_df.data_name.unique().tolist())
    data_name_select = alt.selection_single(fields=['data_name'], bind=data_name_dropdown, name="Data Name", 
                                                init={'data_name': 'train'})

    # A dropdown filter
    model_name_dropdown = alt.binding_select(options=mdm_df.model_name.unique().tolist())
    model_name_select = alt.selection_single(fields=['model_name'], bind=model_name_dropdown, name="Model Name", 
                                                init={'model_name': mdm_df.model_name.unique()[0]})


    def plot_bounds(attr, border=0.05):
        min_v = mdm_df[attr].min()
        max_v = mdm_df[attr].max()
        gap = border * (max_v - min_v) / 2
        return (min_v - gap, max_v + gap)


    base = alt.Chart(mdm_df).mark_circle(size=60).encode(
        x=alt.X(f'{emb_options[emb]}_emb_coord_1', scale=alt.Scale(domain=plot_bounds(f'{emb_options[emb]}_emb_coord_1'))),
        y=alt.Y(f'{emb_options[emb]}_emb_coord_2', scale=alt.Scale(domain=plot_bounds(f'{emb_options[emb]}_emb_coord_2'))),
        color='layer_name'
    ).add_selection(
        select_epoch
    ).transform_filter(
        select_epoch
    )

    base = base.add_selection(
        model_name_select
    ).transform_filter(
        model_name_select
    )

    return base.add_selection(
        data_name_select
    ).transform_filter(
        data_name_select
    ).properties(title=f"{emb} Embedding")


def intrinsic_dimension_plot(mdm_df):
    slider = alt.binding_range(min=0, max=100, step=1)
    select_epoch = alt.selection_single(name="epoch", fields=['epoch'],
                                    bind=slider, init={'epoch': 0})

    # A dropdown filter
    model_name_dropdown = alt.binding_select(options=mdm_df.model_name.unique().tolist())
    model_name_select = alt.selection_single(fields=['model_name'], bind=model_name_dropdown, name="Model Name", 
                                                init={'model_name': mdm_df.model_name.unique()[0]})

    def plot_bounds(attr, border=0.05):
        min_v = mdm_df[attr].min()
        max_v = mdm_df[attr].max()
        gap = border * (max_v - min_v) / 2
        return (0, max_v + gap)

    base = alt.Chart(mdm_df).transform_calculate(
        ymin="datum.intrinsic_dim_mean-datum.intrinsic_dim_err",
        ymax="datum.intrinsic_dim_mean+datum.intrinsic_dim_err"
    )

    int_dim_lineplot = base.mark_line().encode(
        x='layer_name',
        y=alt.Y('intrinsic_dim_mean', axis=alt.Axis(title='intrinsic dimension'), scale=alt.Scale(domain=plot_bounds('intrinsic_dim_mean'))),
        color='data_name'
    )

    error_bars = base.mark_errorbar().encode(
        x='layer_name',
        y=alt.Y('ymin:Q', axis=alt.Axis(title='intrinsic dimension')),
        y2='ymax:Q',
        color='data_name'
    )

    int_dim_circleplot = base.mark_circle(size=60).encode(
        x='layer_name',
        y=alt.Y('intrinsic_dim_mean', axis=alt.Axis(title='intrinsic dimension')),
        color='data_name'
    )

    return (int_dim_lineplot + error_bars + int_dim_circleplot).add_selection(
        select_epoch
    ).transform_filter(
        select_epoch
    ).add_selection(
        model_name_select
    ).transform_filter(
        model_name_select
    )