// comment

function add_params_on_display(gate, params) {
    gate_info = d3.select(".gate-panel")
    gate_info.append("p").html("Params:")

    for (p in params) {
        gate_info.append("p").html(`- ${p}: ${params[p]}`)
    }
}

function create_params_str(params) {
    params_str = ""

    params_str += "("
    for (p in params) {
        params_str += `${params[p]}, `
    }
    params_str = params_str.slice(0, params_str.length - 2)
    params_str += ")"

    return params_str
}

function one_qubit_gate(x, y, gate_name, register, params={}, color="#33C4FF") {
    let width = 0
    let height = 40
    params_str = ""

    if (Object.keys(params).length !== 0) {
        params_str = create_params_str(params)
        width = Math.max(40, 15*2 + gate_name.length*11, 15*2 + params_str.length*5)
    } else {
        width = Math.max(40, 15*2 + gate_name.length*11)
    }
    let gate = d3.select("#circuit-detail-svg").append("g")
        .attr("transform", `translate(${x - width/2}, ${y - height/2})`)

    gate.append("rect")
        .attr("class", gate_name)
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", width)
        .attr("height", 40)
        .style("fill", color)
    g_name = gate.append("text")
        .text(gate_name)
        .attr("textLength", gate_name.length * 11)
        .attr("x", width / 2)
        .attr("y", 0 + 40 / 2)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")

    if (Object.keys(params).length !== 0) {
        g_name.attr("y", -5 + 40 / 2)

        gate.append("text")
        .text(params_str)
        .attr("font-size", "0.8em")
        .attr("textLength", params_str.length * 5)
        .attr("x", width / 2)
        .attr("y", 10 + 40 / 2)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")
    }
    
    // add on display
    div = d3.select(".tooltip")
    gate_info = d3.select(".gate-panel")
    gate.on("mouseover", function(d) {
        div.transition()
            .duration(200)
            .style("opacity", .7);
        div.html(`${gate_name} gate on ${register}`)
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY) + "px");
        gate_info.append("p").html(`${gate_name} gate on ${register}`)
        add_params_on_display(gate, params)
        })
    .on("mouseout", function(d) {
        div.transition()
            .duration(500)
            .style("opacity", 0);
        gate_info.html("Gate Info:")
    });

    return gate
}

function reset(x, y, register) {
    let gate = d3.select("#circuit-detail-svg").append("g")
        .attr("transform", `translate(${x - 20}, ${y - 20})`)

    gate.append("rect")
        .attr("class", "reset")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 40)
        .attr("height", 40)
        .style("fill", "black")
    gate.append("text")
        .text("0")
        .attr("font-size", "1.2em")
        .attr("x", 20)
        .attr("y", 20+2)
        .style("fill", "white")
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")
    gate.append("line")
        .attr("x1", 10)
        .attr("x2", 10)
        .attr("y1", 10)
        .attr("y2", 30)
        .attr("stroke", "white")
        .style("stroke-width", 1)
    gate.append("line")
        .attr("x1", 27)
        .attr("x2", 33)
        .attr("y1", 10)
        .attr("y2", 20)
        .attr("stroke", "white")
        .style("stroke-width", 1)
    gate.append("line")
        .attr("x1", 27)
        .attr("x2", 33)
        .attr("y1", 30)
        .attr("y2", 20)
        .attr("stroke", "white")
        .style("stroke-width", 1)

    return gate
}

function hadamard(x, y, register) {
    gate = one_qubit_gate(x, y, "H", register)

    return gate
}

function x_gate(x, y, register) {
    gate = one_qubit_gate(x, y, "X", register)

    return gate
}

function y_gate(x, y, register) {
    gate = one_qubit_gate(x, y, "Y", register,"green")

    return gate
}

function z_gate(x, y, register) {
    gate = one_qubit_gate(x, y, "Z", register, "red")

    return gate
}

function p_gate(x, y, register) {
    gate = one_qubit_gate(x, y, "P", register, "yellow")

    return gate
}

function rx(x, y, register, params) {
    if (params == null || params == "") {
        params = "(pi/2)"
    }
    let gate = one_qubit_gate(x, y, "RX", register, color="#33C4FF", params=params)

    return gate
}

function ry(x, y, register, params) {
    if (params == null || params == "") {
        params = "(pi/2)"
    }
    let gate = one_qubit_gate(x=x, y=y, gate_name="RY", register, color="#33C4FF", params=params)

    return gate
}

function rz(x, y, register, params) {
    if (params == null || params == "") {
        params = "(pi/2)"
    }
    let gate = one_qubit_gate(x=x, y=y, gate_name="RZ", register, color="#33C4FF", params=params)

    return gate
}


// Two qubit gate
function create_control_at(element, x1, y1, y2, color="#002D9C") {
    element.append("line")
        .attr("x1", x1)
        .attr("x2", x1)
        .attr("y1", y1)
        .attr("y2", y2)
        .attr("stroke", color)
    element.append("circle")
        .attr("cx", x1)
        .attr("cy", y2)
        .attr("r", 5)
        .attr("fill", color)

    return element
}

function cnot(x1, y1, y2, control, target) {
    gate = d3.select("#circuit-detail-svg").append("g")
        .attr("transform", `translate(${x1}, ${y1})`)
    gate_info = d3.select(".gate-panel")
    // draw a circle at (0, 0), then draw two line to form a cross in the middle
    gate.append("circle")
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("r", 20)
        .attr("fill", "#002D9C")
    gate.append("line")
        .attr("x1", -10)
        .attr("x2", 10)
        .attr("y1", 0)
        .attr("y2", 0)
        .attr("stroke", "white")
    gate.append("line")
        .attr("x1", 0)
        .attr("x2", 0)
        .attr("y1", 10)
        .attr("y2", -10)
        .attr("stroke", "white")

    // draw the control gate
    create_control_at(gate, 0, 20, y2)
    // add on display
    div = d3.select(".tooltip")
    gate.on("mouseover", function(d) {
        div.transition()
            .duration(200)
            .style("opacity", .7);
        div.html(`CNOT gate`)
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY) + "px");
        gate_info.html(`Gate Info: CNOT gate, control - ${control}, target - ${target}`)
        })
    .on("mouseout", function(d) {
        div.transition()
            .duration(500)
            .style("opacity", 0);
        gate_info.html("Gate Info")
    });

    return gate
}

function cz(x1, y1, y2) {
    gate = gate = d3.select("#circuit-detail-svg").append("g")
        .attr("transform", `translate(${x1}, ${y1})`)

    // draw a circle at (0, 0), then draw two line to form a cross in the middle
    gate.append("circle")
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("r", 5)
        .attr("fill", "#002D9C")


    // draw the target register
    create_control_at(gate, 0, 0, y2)

    div = d3.select(".tooltip")
    gate.on("mouseover", function(d) {
        div.transition()
            .duration(200)
            .style("opacity", .7);
        div.html(`CZ gate`)
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY) + "px");
        })
    .on("mouseout", function(d) {
        div.transition()
            .duration(500)
            .style("opacity", 0);
    });

    return gate
}

function crx(x1, y1, y2, params="(pi/2)") {
    gate_name = "RX"
    let gate = rx(x1, y1, "q1", params)
    let width = 0
    if (params != null) {
        width = Math.max(40, 15*2 + gate_name.length*12, 15*2 + params.length*5)
    } else {
        width = Math.max(40, 15*2 + gate_name.length*12)
    }

    create_control_at(gate, width/2, 20, y2+20, "#33C4FF")

    return gate
}

function cry(x1, y1, y2, params="(pi/2)") {
    gate_name = "RY"
    let gate = ry(x1, y1, "q1", params)
    let width = 0
    if (params != null) {
        width = Math.max(40, 15*2 + gate_name.length*12, 15*2 + params.length*5)
    } else {
        width = Math.max(40, 15*2 + gate_name.length*12)
    }

    create_control_at(gate, width/2, 20, y2+20, "#33C4FF")

    return gate
}

function crz(x1, y1, y2, params="(pi/2)") {
    gate_name = "RZ"
    let gate = rz(x1, y1, "q1", params)
    let width = 0
    if (params != null) {
        width = Math.max(40, 15*2 + gate_name.length*12, 15*2 + params.length*5)
    } else {
        width = Math.max(40, 15*2 + gate_name.length*12)
    }

    create_control_at(gate, width/2, 20, y2+20, "#33C4FF")

    return gate
}

// draw measurement
function measure(x1, y1, y2, cbit=0) {
    let measure_z = d3.select("#circuit-detail-svg").append("g")
        .attr("transform", `translate(${x1 - 20}, ${y1 - 20})`)

    measure_z.append("rect")
        .attr("class", "measure")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 40)
        .attr("height", 40)
        .style("fill", "gray")
    measure_z.append("path")
        .attr("d", "M 10 30 C 10 15, 30 15, 30 30")
        .attr("stroke", "black")
        .style("fill", "transparent")
        .style("stroke-width", 2)
    measure_z.append("line")
        .attr("x1", 20)
        .attr("y1", 30)
        .attr("x2", 30)
        .attr("y2", 15)
        .attr("stroke", "black")
        .style("stroke-width", 2)
    measure_z.append("text")
        .text("z")
        .attr("font-size", "1em")
        .attr("x", 20)
        .attr("y", 10)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")
    measure_z.append("line")
        .attr("x1", 18)
        .attr("y1", 40)
        .attr("x2", 18)
        .attr("y2", y2+20-7)
        .attr("stroke", "gray")
        .attr("stroke-width", 2)
        .style("fill", "transparent")
    measure_z.append("line")
        .attr("x1", 22)
        .attr("y1", 40)
        .attr("x2", 22)
        .attr("y2", y2+20-7)
        .attr("stroke", "gray")
        .attr("stroke-width", 2)
        .style("fill", "transparent")
    measure_z.append("polygon")
        .attr("points", `13,${y2+20-7} 27,${y2+20-7} 20,${y2+20}`)
        .style("fill", "gray")
        .style("stroke-width", 2)

    return measure_z
}

// draw classical control
function create_classical_control_at(element, x1, y1, y2, color="gray") {
    element.append("line")
        .attr("x1", x1 - 2)
        .attr("x2", x1 - 2)
        .attr("y1", y1)
        .attr("y2", y2)
        .attr("stroke", color)
        .style('stroke-width', 2)
    element.append("line")
        .attr("x1", x1 + 2)
        .attr("x2", x1 + 2)
        .attr("y1", y1)
        .attr("y2", y2)
        .attr("stroke", color)
        .style('stroke-width', 2)
    element.append("circle")
        .attr("cx", x1)
        .attr("cy", y2)
        .attr("r", 5)
        .attr("fill", color)

    return element
}

// draw barrier
function barrier(x, y, register) {
    let b = d3.select("#circuit-detail-svg").append("g")
        .attr("transform", `translate(${x - 20}, ${y - 20})`)

    b.append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 40)
        .attr("height", 40)
        .style("fill", "transparent")
    b.append("line")
        .attr("x1", 20)
        .attr("y1", 0)
        .attr("x2", 20)
        .attr("y2", 40)
        .attr("stroke", "black")
        .style("stroke-width", 2)
        .style("stroke-dasharray", "3,3")

    return b
}