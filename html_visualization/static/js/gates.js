function add_params_on_display(gate, params, controls) {
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

function one_qubit_gate(x, y, gate_name, register, params={}, controls=[], color="#33C4FF") {
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

    // draw controls
    if (controls.length > 0) {
        for (let i = 0; i < controls.length; i++) {
            y2 = visualization_info.registers.qreg[controls[i]]
            create_control_at(gate, x, y, y2, color)
        }
    }

    gate.append("rect")
        .attr("class", gate_name)
        .attr("x", x - width/2)
        .attr("y", y - height/2)
        .attr("width", width)
        .attr("height", 40)
        .style("fill", color)
    g_name = gate.append("text")
        .text(gate_name)
        .attr("textLength", gate_name.length * 11)
        .attr("x", x)
        .attr("y", y)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")

    if (Object.keys(params).length !== 0) {
        g_name.attr("y", y-5)

        gate.append("text")
        .text(params_str)
        .attr("font-size", "0.8em")
        .attr("textLength", params_str.length * 5)
        .attr("x", x)
        .attr("y", y+10)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")
    }
    
    // add on display
//    div = d3.select(".tooltip")
//    gate_info = d3.select(".gate-panel")
//    gate.on("mouseover", function(d) {
//        div.transition()
//            .duration(200)
//            .style("opacity", .7);
//        div.html(`${gate_name} gate on ${register}`)
//            .style("left", (d3.event.pageX) + "px")
//            .style("top", (d3.event.pageY) + "px");
//        gate_info.append("p").html(`${gate_name} gate on ${register}`)
//        add_params_on_display(gate, params)
//        })
//    .on("mouseout", function(d) {
//        div.transition()
//            .duration(500)
//            .style("opacity", 0);
//        gate_info.html("Gate Info:")
//    });

    return gate
}

function reset(x, y, register) {
    let gate = d3.select("#circuit-detail-svg").append("g")

    gate.append("rect")
        .attr("class", "reset")
        .attr("x", x-20)
        .attr("y", y-20)
        .attr("width", 40)
        .attr("height", 40)
        .style("fill", "black")
    gate.append("text")
        .text("0")
        .attr("font-size", "1.2em")
        .attr("x", x)
        .attr("y", y+2)
        .style("fill", "white")
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")
    gate.append("line")
        .attr("x1", x-10)
        .attr("x2", x-10)
        .attr("y1", y-10)
        .attr("y2", y+10)
        .attr("stroke", "white")
        .style("stroke-width", 1)
    gate.append("line")
        .attr("x1", x+7)
        .attr("x2", x+13)
        .attr("y1", y-10)
        .attr("y2", y)
        .attr("stroke", "white")
        .style("stroke-width", 1)
    gate.append("line")
        .attr("x1", x+7)
        .attr("x2", x+13)
        .attr("y1", y+10)
        .attr("y2", y)
        .attr("stroke", "white")
        .style("stroke-width", 1)

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
    gate_info = d3.select(".gate-panel")

    // draw a circle at (0, 0), then draw two line to form a cross in the middle
    create_control_at(gate, x1, y1, y2)
    gate.append("circle")
        .attr("cx", x1)
        .attr("cy", y1)
        .attr("r", 20)
        .attr("fill", "#002D9C")
    gate.append("line")
        .attr("x1", x1-10)
        .attr("x2", x1+10)
        .attr("y1", y1)
        .attr("y2", y1)
        .attr("stroke", "white")
    gate.append("line")
        .attr("x1", x1)
        .attr("x2", x1)
        .attr("y1", y1+10)
        .attr("y2", y1-10)
        .attr("stroke", "white")
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

// draw measurement
function measure(x1, y1, y2, cbit=0) {
    let measure_z = d3.select("#circuit-detail-svg").append("g")
//        .attr("transform", `translate(${x1 - 20}, ${y1 - 20})`)

    measure_z.append("rect")
        .attr("class", "measure")
        .attr("x", x1-20)
        .attr("y", y1-20)
        .attr("width", 40)
        .attr("height", 40)
        .style("fill", "gray")
    measure_z.append("path")
        .attr("d", `M ${x1-10} ${y1+10} C ${x1-10} ${y1-5}, ${x1+10} ${y1-5}, ${x1+10} ${y1+10}`) //M 10 30 C 10 15, 30 15, 30 30
        .attr("stroke", "black")
        .style("fill", "transparent")
        .style("stroke-width", 2)
    measure_z.append("line")
        .attr("x1", x1)
        .attr("y1", y1+10)
        .attr("x2", x1+10)
        .attr("y2", y1-5)
        .attr("stroke", "black")
        .style("stroke-width", 2)
    measure_z.append("text")
        .text("z")
        .attr("font-size", "1em")
        .attr("x", x1)
        .attr("y", y1-10)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")
    measure_z.append("line")
        .attr("x1", x1-2)
        .attr("y1", y1+20)
        .attr("x2", x1-2)
        .attr("y2", y2-7)
        .attr("stroke", "gray")
        .attr("stroke-width", 2)
        .style("fill", "transparent")
    measure_z.append("line")
        .attr("x1", x1+2)
        .attr("y1", y1+20)
        .attr("x2", x1+2)
        .attr("y2", y2-7)
        .attr("stroke", "gray")
        .attr("stroke-width", 2)
        .style("fill", "transparent")
    measure_z.append("polygon")
        .attr("points", `${x1-7},${y2-7} ${x1+7},${y2-7} ${x1},${y2}`)
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
function barrier(x, y) {
    let b = d3.select("#circuit-detail-svg").append("g")

    b.append("rect")
        .attr("x", x)
        .attr("y", y)
        .attr("width", 40)
        .attr("height", 40)
        .style("fill", "transparent")
    b.append("line")
        .attr("x1", x)
        .attr("y1", y - 20)
        .attr("x2", x)
        .attr("y2", y + 20)
        .attr("stroke", "black")
        .style("stroke-width", 2)
        .style("stroke-dasharray", "3,3")
    return b
}