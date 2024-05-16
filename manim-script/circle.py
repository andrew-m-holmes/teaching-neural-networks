import manim


class CreateCircle(manim.Scene):
    def construct(self):
        circle = manim.Circle()
        circle.set_fill(manim.RED, opacity=0.75)
        self.play(manim.Create(circle))
