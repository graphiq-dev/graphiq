// comment

function one_qubit_gate(x, y, gate_name, color="#33C4FF", params=null) {
    let width = 0
    let height = 40
    if (params != null) {
        width = Math.max(40, 15*2 + gate_name.length*10, 15*2 + params.length*5)
    } else {
        width = Math.max(40, 15*2 + gate_name.length*10)
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
    if (params == null) {
        gate.append("text")
        .text(gate_name)
        .attr("letter-spacing", 1)
        .attr("font-size", "1em")
        .attr("textLength", gate_name.length * 10)
        .attr("x", width / 2)
        .attr("y", 0 + 40 / 2)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")
    } else {
        gate.append("text")
        .text(gate_name)
        .attr("letter-spacing", 1)
        .attr("font-size", "1em")
        .attr("textLength", gate_name.length * 10)
        .attr("x", width / 2)
        .attr("y", -5 + 40 / 2)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")

        gate.append("text")
        .text(params)
        .attr("font-size", "0.8em")
        .attr("textLength", params.length * 5)
        .attr("x", width / 2)
        .attr("y", 10 + 40 / 2)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")
    }
    
    // add on display
    return gate
}

function hadamard(x, y, register) {
    gate = one_qubit_gate(x, y, "H")
    
    // add on display
    div = d3.select(".tooltip")
    gate_info = d3.select(".gate-panel")
    gate.on("mouseover", function(d) {		
        div.transition()		
            .duration(200)		
            .style("opacity", .7);		
        div.html(`H gate on ${register}`)	
            .style("left", (d3.event.pageX) + "px")		
            .style("top", (d3.event.pageY) + "px");
        gate_info.html(`Gate Info: Hadamard gate on ${register}`)
        })					
    .on("mouseout", function(d) {		
        div.transition()		
            .duration(500)		
            .style("opacity", 0);
        gate_info.html("Gate Info:")
    });

    return gate
}

function x_gate(x, y, register) {
    gate = one_qubit_gate(x, y, "X")
    
    // add on display
    gate_info = d3.select(".gate-panel")

    div = d3.select(".tooltip")
    gate.on("mouseover", function(d) {		
        div.transition()		
            .duration(200)		
            .style("opacity", .7);		
        div.html(`X gate on ${register}`)	
            .style("left", (d3.event.pageX) + "px")		
            .style("top", (d3.event.pageY) + "px");
        gate_info.html(`Gate Info: X gate on ${register}`)
        })
    .on("mouseout", function(d) {		
        div.transition()		
            .duration(500)		
            .style("opacity", 0);
        gate_info.html("Gate Info:")
    });

    return gate
}

function y_gate(x, y, register) {
    gate = one_qubit_gate(x=x, y=y, gate_name="Y", color="green")

    // add on display
    div = d3.select(".tooltip")
    gate_info = d3.select(".gate-panel")
    gate.on("mouseover", function(d) {
        div.transition()
            .duration(200)
            .style("opacity", .7);
        div.html(`Y gate on ${register}`)
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY) + "px");
        gate_info.html(`Gate Info: Y gate on ${register}`)
        })
    .on("mouseout", function(d) {
        div.transition()
            .duration(500)
            .style("opacity", 0);
        gate_info.html("Gate Info:")
    });

    return gate
}

function z_gate(x, y, register) {
    gate = one_qubit_gate(x, y, "Z", color="red")

    // add on display
    div = d3.select(".tooltip")
    gate_info = d3.select(".gate-panel")
    gate.on("mouseover", function(d) {
        div.transition()
            .duration(200)
            .style("opacity", .7);
        div.html(`Z gate on ${register}`)
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY) + "px");
        gate_info.html(`Gate Info: Z gate on ${register}`)
        })
    .on("mouseout", function(d) {
        div.transition()
            .duration(500)
            .style("opacity", 0);
        gate_info.html("Gate Info:")
    });

    return gate
}

function p_gate(x, y, register) {
    gate = one_qubit_gate(x, y, "P", color="yellow")

    // add on display
    div = d3.select(".tooltip")
    gate_info = d3.select(".gate-panel")
    gate.on("mouseover", function(d) {
        div.transition()
            .duration(200)
            .style("opacity", .7);
        div.html(`P gate on ${register}`)
            .style("left", (d3.event.pageX) + "px")
            .style("top", (d3.event.pageY) + "px");
        gate_info.html(`Gate Info: P gate on ${register}`)
        })
    .on("mouseout", function(d) {
        div.transition()
            .duration(500)
            .style("opacity", 0);
        gate_info.html("Gate Info:")
    });

    return gate
}

function rx(x, y, register, params) {
    if (params == null || params == "") {
        params = "(pi/2)"
    }
    let gate = one_qubit_gate(x=x, y=y, gate_name="RX", color="#33C4FF", params=params)

    // add display effect

    return gate
}

function ry(x, y, register, params) {
    if (params == null || params == "") {
        params = "(pi/2)"
    }
    let gate = one_qubit_gate(x=x, y=y, gate_name="RY", color="#33C4FF", params=params)

    // add display effect

    return gate
}

function rz(x, y, register, params) {
    if (params == null || params == "") {
        params = "(pi/2)"
    }
    let gate = one_qubit_gate(x=x, y=y, gate_name="RZ", color="#33C4FF", params=params)

    // add display effect

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

    return g
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

    create_control_at(gate, width/2, 0, y2+20, "#33C4FF")

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

    create_control_at(gate, width/2, 0, y2+20, "#33C4FF")

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

    create_control_at(gate, width/2, 0, y2+20, "#33C4FF")

    return gate
}