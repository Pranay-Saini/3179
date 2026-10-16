const embedOptions = {
    actions: false,
    renderer: "canvas"
  };
  
  const specs = [
    { id: "#vis01", file: "js/v01_point_map.vg.json" },
    { id: "#vis02", file: "js/v02_choropleth_lga.vg.json" },
    { id: "#vis03", file: "js/v03_lollipop_seifa.vg.json" },
    { id: "#vis04", file: "js/v04_butterfly_metro_reg.vg.json" },
    { id: "#vis05", file: "js/v05_treemap_surfaces.vg.json" },
    { id: "#vis06", file: "js/v06_grouped_bar_age.vg.json" },
    { id: "#vis07", file: "js/v07_heatmap_access.vg.json" },
    { id: "#vis08", file: "js/v08_waffle_national.vg.json" },
    { id: "#vis09", file: "js/v09_buffer_catchment.vg.json" },
    { id: "#vis10", file: "js/v10_scatter_grants.vg.json" },
    { id: "#vis11", file: "js/v11_jitter_rounds.vg.json" },
    { id: "#vis12", file: "js/v12_slope_growth.vg.json" }
  ];
  
  specs.forEach(item => {
    vegaEmbed(item.id, item.file, embedOptions)
      .then(function(result) {
        // Vega view instance initialized
      })
      .catch(console.error);
  });