// this file is used to draw on html

function draw_register_info(register) {
    qreg = register["qreg"]
    creg = register["creg"]

    for (reg in qreg) {
        draw_register_label(reg, 70, qreg[reg])
        draw_quantum_register(0, qreg[reg])
    }

    for (reg in creg) {
        draw_register_label(reg, 70, creg[reg])
        draw_classical_register(0, creg[reg])
    }
}

function draw_gate_info(gates) {
    for (let i = 0; i < gates.length; i++) {
        // one_qubit_gate(x, y, gate_name, register, color="#33C4FF", params=null)
        register = visualization_info.register.qreg[gates[i].qarg]

        g = one_qubit_gate(gates[i].x_pos, register, gates[i].gate_name, gates[i].qarg)

        if (gates[i].control != "") {
            color = g.select("rect").style("fill")

            x = g.select("rect").attr("width") / 2
            y1 = g.select("rect").attr("height")
            y2 = visualization_info.register.qreg[gates[i].control] - register + 20
            // create_control_at(element, x1, y1, y2, color="#002D9C")
            create_control_at(g, x, y1, y2, color)
        }
    }
}