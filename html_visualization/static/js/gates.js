// comment

function one_qubit_gate(x, y, gate_name, params=null, color="#33C4FF") {
    gate = d3.select("#circuit-detail-svg").append("g")
        .attr("transform", `translate(${x - 20}, ${y - 20})`)

    gate.append("rect")
        .attr("class", gate_name)
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 40)
        .attr("height", 40)
        .style("fill", color)
    gate.append("text")
        .text(gate_name)
        .attr("textLength", 10)
        .attr("x", 0 + 40 / 2)
        .attr("y", 0 + 40 / 2)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")
    
    // add on display
    return gate
}

function hadamard(x, y, register) {
    gate = one_qubit_gate(x, y, "H")
    
    // add on display
    div = d3.select(".tooltip")
    gate.on("mouseover", function(d) {		
        div.transition()		
            .duration(200)		
            .style("opacity", .7);		
        div.html(`H gate on ${register}`)	
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
    gate = one_qubit_gate(x, y, "Y", color="green")

    // add on display
    div = d3.select(".tooltip")
    gate.on("mouseover", function(d) {
        div.transition()
            .duration(200)
            .style("opacity", .7);
        div.html(`Y gate on ${register}`)
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

function z_gate(x, y, register) {
    gate = one_qubit_gate(x, y, "Z", color="red")

    // add on display
    div = d3.select(".tooltip")
    gate.on("mouseover", function(d) {
        div.transition()
            .duration(200)
            .style("opacity", .7);
        div.html(`Z gate on ${register}`)
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

function p_gate(x, y, register) {
    gate = one_qubit_gate(x, y, "P", color="yellow")

    // add on display
    div = d3.select(".tooltip")
    gate.on("mouseover", function(d) {
        div.transition()
            .duration(200)
            .style("opacity", .7);
        div.html(`P gate on ${register}`)
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

function cnot(x1, y1, y2) {
    gate = d3.select("#circuit-detail-svg").append("g")
        .attr("transform", `translate(${x1}, ${y1})`)

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
    create_control_at(gate, 0, -20, y2)
    // add on display
    div = d3.select(".tooltip")
    gate.on("mouseover", function(d) {
        div.transition()
            .duration(200)
            .style("opacity", .7);
        div.html(`CNOT gate`)
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