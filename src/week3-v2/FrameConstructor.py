import time


class FrameConstructor:

    @staticmethod
    def _zero_pos_rot(rotation):
        return {
            "position": [0, 0, 0],
            "rotation": [float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])]
        }

    @staticmethod
    def build_frame_from_rotations(rotations, hand="left"):
        """
        Build a frame matching HandDataGenerator.py's format from a list of 21 quaternions.

        rotations: list[21] of [x, y, z, w] quaternions corresponding to MediaPipe landmarks indices 0..20
        hand: "left" or "right"
        """

        if rotations is None or len(rotations) != 21:
            raise ValueError("rotations must be a list of length 21")

        # MediaPipe indices mapping to bones
        # 0: wrist
        # Thumb: 1(CMC),2(MCP),3(IP),4(Tip)
        # Index: 5(MCP),6(PIP),7(DIP),8(Tip)
        # Middle: 9(MCP),10(PIP),11(DIP),12(Tip)
        # Ring: 13(MCP),14(PIP),15(DIP),16(Tip)
        # Pinky: 17(MCP),18(PIP),19(DIP),20(Tip)

        frame = {
            "timestamp": time.time(),
            "hand": hand,
            "wrist": {
                "position": [0, 0, 0],
                "rotation": [
                    float(rotations[0][0]),
                    float(rotations[0][1]),
                    float(rotations[0][2]),
                    float(rotations[0][3])
                ]
            },
            "thumb": {
                "metacarpal": FrameConstructor._zero_pos_rot(rotations[1]),
                "proximal": FrameConstructor._zero_pos_rot(rotations[2]),
                "intermediate": FrameConstructor._zero_pos_rot(rotations[3]),
                "distal": FrameConstructor._zero_pos_rot(rotations[4])
            },
            "index": {
                "metacarpal": FrameConstructor._zero_pos_rot(rotations[5]),
                "proximal": FrameConstructor._zero_pos_rot(rotations[6]),
                "intermediate": FrameConstructor._zero_pos_rot(rotations[7]),
                "distal": FrameConstructor._zero_pos_rot(rotations[8])
            },
            "middle": {
                "metacarpal": FrameConstructor._zero_pos_rot(rotations[9]),
                "proximal": FrameConstructor._zero_pos_rot(rotations[10]),
                "intermediate": FrameConstructor._zero_pos_rot(rotations[11]),
                "distal": FrameConstructor._zero_pos_rot(rotations[12])
            },
            "ring": {
                "metacarpal": FrameConstructor._zero_pos_rot(rotations[13]),
                "proximal": FrameConstructor._zero_pos_rot(rotations[14]),
                "intermediate": FrameConstructor._zero_pos_rot(rotations[15]),
                "distal": FrameConstructor._zero_pos_rot(rotations[16])
            },
            "pinky": {
                "metacarpal": FrameConstructor._zero_pos_rot(rotations[17]),
                "proximal": FrameConstructor._zero_pos_rot(rotations[18]),
                "intermediate": FrameConstructor._zero_pos_rot(rotations[19]),
                "distal": FrameConstructor._zero_pos_rot(rotations[20])
            }
        }

        return frame


