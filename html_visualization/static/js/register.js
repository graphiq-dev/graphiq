// Note: Svg draw function has return line for testing. 

// Static variables
// state variable to keep track of the state of the web app. Maybe use redux for state management ?
// TODO: consider have a styles variable to store all styles ?. 
let next_reg_label = 0
let next_reg_line = 0
let state = {
    register_position: {},
}



// init function
function register_label_init() {
    const register_label_container = d3.select("body")
        .append("div")
        .attr("class", "circuit-label-container")
    register_label_container.append("svg")
        .attr("id", "circuit-label-svg")
        .attr("width", 70)
        .attr("height", 400)
    
    
    return register_label_container
}

function register_detail_init() {
    const register_detail_container = d3.select("body")
        .append("div")
        .attr("class", "circuit-detail-container")
    register_detail_container.append("svg")
        .attr("id", "circuit-detail-svg")
        .attr("width", 3000)
        .attr("height", 400)

    return register_detail_container
}


// draw function
// draw the quantum register
function draw_quantum_register(name, reg_num) {
    next_reg_line++
    state.register_position[`${name}${reg_num}`] = next_reg_line * 50

    const register = d3.select("#circuit-detail-svg")
    const g = register.append("g")

    g.append('line')
        .style("stroke", "black")
        .style("fill", "none")
        .attr("x1", 0)
        .attr("y1", next_reg_line * 50)
        .attr("x2", register.attr("width"))
        .attr("y2", next_reg_line * 50)
        .attr("name", `${name}[${reg_num}]`);

    return register
}

// draw the classical register
function draw_classical_register(name, reg_num) {
    next_reg_line++
    state.register_position[`${name}${reg_num}`] = next_reg_line * 50

    const register = d3.select("#circuit-detail-svg")
    const g = register.append("g")

    g.append('line')
        .style("stroke", "black")
        .attr("x1", 0)
        .attr("y1", next_reg_line * 50 - 1)
        .attr("x2", register.attr("width"))
        .attr("y2", next_reg_line * 50 - 1)
        .attr("name", `${name}[${reg_num}]`);
    g.append('line')
        .style("stroke", "black")
        .attr("x1", 0)
        .attr("y1", next_reg_line * 50 + 2)
        .attr("x2", register.attr("width"))
        .attr("y2", next_reg_line * 50 + 2)
        .attr("name", `${name}[${reg_num}]`);
    
    return register
}

// draw register label for both quantum and classical register
function draw_register_label(reg_type, name, reg_num) {
    x = 70
    next_reg_label++
    y = next_reg_label * 50
    const new_register_label = d3.select("#circuit-label-svg")

    graph = new_register_label.append("g")
        .attr("id", `${reg_type}-${name}-${reg_num}`)

    text = graph.append("text")
        .text(`${name}[${reg_num}]`)
        .attr("class", "register-label")
        .attr("x", x)
        .attr("y", y)
        .attr("dy", ".3em")

    return new_register_label
}