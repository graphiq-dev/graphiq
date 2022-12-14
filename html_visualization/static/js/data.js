//let visualization_info = {
//    registers: {
//        qreg: {p0: 50, p1: 100, p2: 150, p3: 200, e0: 250},
//        creg: {c4: 300}
//    },
//    gates: [
//        {
//            gate_name: "H",
//            qargs: ["e0"],
//            controls: [],
//            params: {},
//            x_pos: 50,
//        },
//        {
//            gate_name: "CX",
//            qargs: ["p1"],
//            controls: ["e0"],
//            params: {},
//            x_pos: 100,
//        },
//        {
//            gate_name: "U",
//            qargs: ["p2"],
//            controls: ["e0"],
//            params: {
//                theta: "pi/2",
//                phi: "pi/4",
//                lambda: "pi/8",
//            },
//            x_pos: 200,
//        }
//    ],
//    measurements: [
//        {
//            x_pos: 300,
//            qreg: "p0",
//            creg: "c4",
//            cbit: 0,
//        },
//        {
//            x_pos: 350,
//            qreg: "p1",
//            creg: "c4",
//            cbit: 0,
//        },
//        {
//            x_pos: 400,
//            qreg: "p2",
//            creg: "c4",
//            cbit: 0,
//        },
//        {
//            x_pos: 450,
//            qreg: "p3",
//            creg: "c4",
//            cbit: 0,
//        },
//    ]
//}