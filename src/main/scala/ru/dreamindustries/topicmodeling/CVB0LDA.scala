package ru.dreamindustries.topicmodeling

import util.Random

object CVB0LDA {
  def estimateTopicAssignments[TW](docs: Seq[Seq[TW]], alpha: Seq[Double], beta: Double) = {
    val K = alpha.size
    val words = docs.flatten.toSet
    val V = words.size

    val q_z = docs.map { doc =>
      doc.map { word =>
        val d = Array.fill(K) { Random.nextDouble() }
        val normalizer = d.sum
        d.map(_ / normalizer)
      }
    }

    val all = (for (d <- 0 to docs.size;
      i <- 0 to docs(d).size) yield (d, i)).par

    def N_t_w(t: Int, w: TW) =
      all.filter{ case (d, i) => docs(d)(i) == w }.map{ case (d, i) => q_z(d)(i)(t) }.sum

    def N_d_t(d: Int, t: Int) =
      (0 to docs(d).size).map(i => q_z(d)(i)(t)).sum

    def variationalUpdate(d: Int, i: Int) = {
      val w = docs(d)(i)

      val new_d = Array.fill(K) {
        0.0
      }
      val normalizer = new_d.sum

      q_z(d).updated(i, new_d.map(_ / normalizer))
    }

    q_z
  }
}
