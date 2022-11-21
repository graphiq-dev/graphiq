// comment

function hadamard(x, y) {
    gate = d3.select("#circuit-detail-svg").append("g")
        .attr("transform", `translate(${x}, ${y})`)

    gate.append("rect")
        .attr("class", "hadamard")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 40)
        .attr("height", 40)
        .style("fill", "#33C4FF")
    gate.append("text")
        .text("H")
        .attr("textLength", 10)
        .attr("x", 0 + 40 / 2)
        .attr("y", 0 + 40 / 2)
        .style("text-anchor", "middle")
        .style("dominant-baseline", "middle")
    
    // add on display
    div = d3.select(".tooltip")
    gate.on("mouseover", function(d) {		
        div.transition()		
            .duration(200)		
            .style("opacity", .7);		
        div.html(`Hadamard, register`)	
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

function cnot(x, y) {
    gate = gate = d3.select("#circuit-detail-svg").append("g")
        .attr("transform", `translate(${x}, ${y})`)

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
    gate.append("line")
        .attr("x1", 0)
        .attr("x2", 0)
        .attr("y1", -20)
        .attr("y2", -100)
        .attr("stroke", "#002D9C")
    gate.append("circle")
        .attr("cx", 0)
        .attr("cy", -100)
        .attr("r", 5)
        .attr("fill", "#002D9C")

    return gate
}