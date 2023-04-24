/**
 * Function to draw register label initial area.
 * @param {number} width - The width of the register label area
 * @param {number} height - The height of the register label area
 * @return {Object} - Function return the container of the register label.
 */
function register_label_init(width, height) {
    const register_label_container = d3.select("div.circuit-container")
        .append("div")
        .attr("class", "circuit-label-container")
    register_label_container.append("svg")
        .attr("id", "circuit-label-svg")
        .attr("width", width)
        .attr("height", height)
    
    return register_label_container
}

/**
 * Function to draw register detail initial area.
 * @param {number} width - The width of the register detail area
 * @param {number} height - The height of the register detail area
 * @return {Object} - Function return the container of the register detail.
 */
function register_detail_init(width, height) {
    const register_detail_container = d3.select("div.circuit-container")
        .append("div")
        .attr("class", "circuit-detail-container")
    svg = register_detail_container.append("svg")
        .attr("id", "circuit-detail-svg")
        .attr("width", width)
        .attr("height", height)
    let zoom = d3.zoom()
      .on('zoom', handleZoom);

    function handleZoom(e) {
      d3.selectAll('#circuit-detail-svg g')
        .attr('transform', e.transform);
    }

    svg.call(zoom);

    return register_detail_container
}


/**
 * Function to draw quantum register. A single line that have the same width with the register detail area.
 * @param {number} x - The x position of the register
 * @param {number} y - The y position of the register
 * @param {number} width - The width of the register
 * @return {Object} - Function return the quantum register object.
 */
function draw_quantum_register(x, y, width) {
    const register = d3.select("#circuit-detail-svg")
    g = register.append("g")

    g.append('line')
        .style("stroke", "black")
        .style("fill", "none")
        .attr("x1", x)
        .attr("y1", y)
        .attr("x2", width)
        .attr("y2", y)

    return register
}

/**
 * Function to draw classical register. A double lines that have the same width with the register detail area.
 * @param {number} x - The x position of the register
 * @param {number} y - The y position of the register
 * @param {number} width - The width of the register
 * @return {Object} - Function return the classical register object.
 */
function draw_classical_register(x, y, width) {
    const register = d3.select("#circuit-detail-svg")
    const g = register.append("g")

    g.append('line')
        .style("stroke", "black")
        .attr("x1", x)
        .attr("y1", y-1)
        .attr("x2", width)
        .attr("y2", y-1)
    g.append('line')
        .style("stroke", "black")
        .attr("x1", x)
        .attr("y1", y+2)
        .attr("x2", width)
        .attr("y2", y+2)
    
    return register
}

/**
 * Function to register label.
 * @param {string} label_name - The name of the register
 * @param {number} x - The x position of the register
 * @param {number} y - The y position of the register
 * @return {Object} - Function return the register label object.
 */
function draw_register_label(label_name, x, y) {
    const new_register_label = d3.select("#circuit-detail-svg")
    graph = new_register_label.append("g")
        .attr("id", `${label_name}`)

    text = graph.append("text")
        .text(`${label_name}`)
        .attr("class", "register-label")
        .attr("x", x)
        .attr("y", y)
        .style("text-anchor", "end")
        .style("dominant-baseline", "middle")

    return new_register_label
}