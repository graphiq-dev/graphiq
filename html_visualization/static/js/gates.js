const std_gate_width = 40
const std_gate_height = 40

/**
 * Function to add gate info to the gate info panel. This panel is on the right of the circuit visualization. When the
 * user hover on a gate the info will be display on this panel.
 * @param  {Object} gate - gate object that will be draw.
 * @param {string} gate_name - name of the gate
 * @param {string} register - name of the register
 * @param {Object} params - params of the gate, only for parametrized gate.
 * @param {Array} controls - all the controls of the gate
 * @return {void} - Function returns nothing
 */
function add_to_gate_info_panel(gate, gate_name, register, params, controls) {
    gate_info = d3.select(".gate-panel")

    gate_info = d3.select(".gate-panel")
    gate.on("mouseover", function(d) {
        gate_info.append("p").html(`${gate_name} gate on ${register}`)
        gate_info.append("p").html("Params:")
        for (p in params) {
            gate_info.append("p").html(`- ${p}: ${params[p]}`)
        }

        gate_info.append("p").html("Control:")
        for (let i = 0; i < controls.length; i++) {
            gate_info.append("p").html(`- ${controls[i]}`)
        }
        })
    .on("mouseout", function(d) {
        div.transition()
            .duration(500)
            .style("opacity", 0);
        gate_info.html("Gate Info:")
    });
}

/**
 * Function to create params string to put on the parametrized gate. The format will be (param1, param2, ...)
 * Ex: (pi/2, pi/2, pi/2)
 * @param  {Object} params - The params object
 * @return {string} - The params string
 */
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

/**
 * Function to draw one qubit gate.
 * @param {number} x - x position
 * @param {number} y - y position
 * @param {string} gate_name - name of the gate
 * @param {string} register - name of register
 * @param {Object} params - params object
 * @param {Array} controls - all the controls of the gate
 * @param {string} color - color of the gate, default is light blue.
 * @return {Object} - Function returns gate object
 */
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
    
    // add gate info to display on panel
    add_to_gate_info_panel(gate, gate_name, register, params, controls)

    return gate
}

/**
 * Function to draw reset operation. Draw |0> inside a rect element, put the object at x, y position on the svg element
 * @param {number} x - x position
 * @param {number} y - y position
 * @param {string} register - name of register
 * @return {Object} - Function returns reset object
 */
function reset(x, y, register) {
    let gate = d3.select("#circuit-detail-svg").append("g")

    gate.append("rect")
        .attr("class", "reset")
        .attr("x", x-20)
        .attr("y", y-20)
        .attr("width", std_gate_width)
        .attr("height", std_gate_height)
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

/**
 * Function to draw control operation. Draw a circle at control position. Draw a line from the circle to the target
 * position
 * @param {Object} element - element to add control to
 * @param {number} x1 - x position of the circle and the line
 * @param {number} y1 - y position of the target position
 * @param {number} y2 - y position of the circle position (control position)
 * @param {string} color - color of the control
 * @return {Object} - Function returns the element to create control at.
 */
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

/**
 * Function to draw cnot gate. Draw X gate at the target position, then draw control at control position.
 * @param {number} x1 - x position of the gate
 * @param {number} y1 - y position of the target position
 * @param {number} y2 - y position of the control position
 * @param {string} control - name of the control register
 * @param {string} target - name of the target register
 * @return {Object} - Function returns the cnot gate object.
 */
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
    add_to_gate_info_panel(gate, "CX", target, {}, control)
    return gate
}

/**
 * Function to draw cz gate.
 * @param {number} x1 - x position of the gate
 * @param {number} y1 - y position of the target position
 * @param {number} y2 - y position of the control position
 * @param {string} control - name of the control register
 * @param {string} target - name of the target register
 * @return {Object} - Function returns cz gate object.
 */
function cz(x1, y1, y2, control, target) {
    gate = gate = d3.select("#circuit-detail-svg").append("g")

    create_control_at(gate, x1, y1, y2)
    // draw a circle at (0, 0), then draw two line to form a cross in the middle
    gate.append("circle")
        .attr("cx", x1)
        .attr("cy", y1)
        .attr("r", 5)
        .attr("fill", "#002D9C")

    // draw the target register
    add_to_gate_info_panel(gate, "CZ", target, {}, control)

    return gate
}

/**
 * Function to draw measurement operation.
 * @param {number} x1 - x position of the operation
 * @param {number} y1 - y position of the quantum register
 * @param {number} y2 - y position of the classical register
 * @param {string} cbit - bit number of the classical reegister
 * @return {Object} - Function returns measurement operation object.
 */
function measure(x1, y1, y2, cbit=0) {
    let measure_z = d3.select("#circuit-detail-svg").append("g")

    measure_z.append("rect")
        .attr("class", "measure")
        .attr("x", x1-20)
        .attr("y", y1-20)
        .attr("width", std_gate_width)
        .attr("height", std_gate_height)
        .style("fill", "gray")
    //M 10 30 C 10 15, 30 15, 30 30
    measure_z.append("path")
        .attr("d", `M ${x1-10} ${y1+10} C ${x1-10} ${y1-5}, ${x1+10} ${y1-5}, ${x1+10} ${y1+10}`)
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

/**
 * Function to create classical control.
 * @param {Object} element - The gate or operation that the classical control will perform.
 * @param {number} x1 - x position of the operation
 * @param {number} y1 - y position of the operation
 * @param {number} y2 - y position of the classical register
 * @param {string} color - color of the classical control
 * @return {Object} - Function returns element that have the classical control.
 */
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

/**
 * Function to draw barrier.
 * @param {number} x - x position of the barrier
 * @param {number} y - y position of the barrier
 * @return {Object} - Function returns barrier object.
 */
function barrier(x, y) {
    let b = d3.select("#circuit-detail-svg").append("g")

    b.append("rect")
        .attr("x", x)
        .attr("y", y)
        .attr("width", std_gate_width)
        .attr("height", std_gate_height)
        .style("fill", "transparent")
    b.append("line")
        .attr("x1", x)
        .attr("y1", y - (std_gate_width/2))
        .attr("x2", x)
        .attr("y2", y + (std_gate_height/2))
        .attr("stroke", "black")
        .style("stroke-width", 2)
        .style("stroke-dasharray", "3,3")
    return b
}