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
function register_label_init(width, height) {
    const register_label_container = d3.select("body")
        .append("div")
        .attr("class", "circuit-label-container")
    register_label_container.append("svg")
        .attr("id", "circuit-label-svg")
        .attr("width", width)
        .attr("height", height)
//        .attr("width", 70)
//        .attr("height", 400)
    
    
    return register_label_container
}

function register_detail_init(width, height) {
    const register_detail_container = d3.select("body")
        .append("div")
        .attr("class", "circuit-detail-container")
    register_detail_container.append("svg")
        .attr("id", "circuit-detail-svg")
        .attr("width", width)
        .attr("height", height)

    return register_detail_container
}


// draw function
// draw the quantum register
function draw_quantum_register(x, y) {
    const register = d3.select("#circuit-detail-svg")

    g = register.append("g")
        .attr("transform", `translate(${x}, ${y})`)

    g.append('line')
        .style("stroke", "black")
        .style("fill", "none")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", register.attr("width"))
        .attr("y2", 0)
        // .attr("name", `${name}[${reg_num}]`);

    return register
}

// draw the classical register
function draw_classical_register(x, y) {
    const register = d3.select("#circuit-detail-svg")
    const g = register.append("g")
        .attr("transform", `translate(${x}, ${y})`)

    g.append('line')
        .style("stroke", "black")
        .attr("x1", 0)
        .attr("y1", next_reg_line * 50 - 1)
        .attr("x2", register.attr("width"))
        .attr("y2", next_reg_line * 50 - 1)
        // .attr("name", `${name}[${reg_num}]`);
    g.append('line')
        .style("stroke", "black")
        .attr("x1", 0)
        .attr("y1", next_reg_line * 50 + 2)
        .attr("x2", register.attr("width"))
        .attr("y2", next_reg_line * 50 + 2)
        // .attr("name", `${name}[${reg_num}]`);
    
    return register
}

// draw register label for both quantum and classical register
function draw_register_label(label_name, x, y) {
//    x = 70
//    next_reg_label++
//    y = next_reg_label * 50
    const new_register_label = d3.select("#circuit-label-svg")

    graph = new_register_label.append("g")
        .attr("id", `${label_name}`)

    text = graph.append("text")
        .text(`${label_name}`)
        .attr("class", "register-label")
        .attr("x", x)
        .attr("y", y)
        .attr("dy", ".3em")

    return new_register_label
}